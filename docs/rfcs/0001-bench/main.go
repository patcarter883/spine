// Beads multi-writer throughput probe for RFC 0001 §9.2.
package main

import (
	"context"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	beads "github.com/steveyegge/beads"
	_ "modernc.org/sqlite"
)

type store interface {
	create(ctx context.Context, title string) (string, error)
	update(ctx context.Context, id, notes string) error
	cas(ctx context.Context, id, notes string) (bool, error) // read version, conditional write
	close()
}

// ---- beads ----
type bstore struct{ s beads.Storage }

func (b *bstore) create(ctx context.Context, t string) (string, error) {
	is := &beads.Issue{Title: t, Status: beads.StatusOpen, IssueType: beads.TypeTask, Priority: 2}
	err := b.s.CreateIssue(ctx, is, "bench")
	return is.ID, err
}
func (b *bstore) update(ctx context.Context, id, n string) error {
	return b.s.UpdateIssue(ctx, id, map[string]interface{}{"notes": n}, "bench")
}
func (b *bstore) cas(ctx context.Context, id, n string) (bool, error) {
	is, err := b.s.GetIssue(ctx, id)
	if err != nil {
		return false, err
	}
	v := is.RowVersion
	// status flips so RowVersion actually changes on each write (notes-only writes do not bump it).
	st := beads.StatusInProgress
	if is.Status == beads.StatusInProgress {
		st = beads.StatusOpen
	}
	err = b.s.UpdateIssueChecked(ctx, id, map[string]interface{}{"notes": n, "status": string(st)}, "bench", beads.UpdateIssueOptions{ExpectedVersion: &v})
	if errors.Is(err, beads.ErrVersionMismatch) {
		return false, nil
	}
	return err == nil, err
}
func (b *bstore) close() { b.s.Close() }

// ---- sqlite (the RFC's proposed store shape) ----
type sstore struct {
	w, r *sql.DB
	mu   sync.Mutex // single writer, as in RFC §7
	seq  atomic.Int64
}

func openSQLite(dir string) (*sstore, error) {
	p := filepath.Join(dir, "spine.db")
	dsn := "file:" + p + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=synchronous(FULL)&_txlock=immediate"
	w, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	w.SetMaxOpenConns(1)
	r, _ := sql.Open("sqlite", dsn)
	r.SetMaxOpenConns(8)
	_, err = w.Exec(`CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, title TEXT, status_json TEXT, rv INTEGER NOT NULL)`)
	return &sstore{w: w, r: r}, err
}
func (s *sstore) create(ctx context.Context, t string) (string, error) {
	id := fmt.Sprintf("t-%d", s.seq.Add(1))
	_, err := s.w.ExecContext(ctx, `INSERT INTO tasks VALUES(?,?,?,0)`, id, t, "{}")
	return id, err
}
func (s *sstore) update(ctx context.Context, id, n string) error {
	_, err := s.w.ExecContext(ctx, `UPDATE tasks SET status_json=?, rv=rv+1 WHERE id=?`, n, id)
	return err
}
func (s *sstore) cas(ctx context.Context, id, n string) (bool, error) {
	var rv int64
	if err := s.r.QueryRowContext(ctx, `SELECT rv FROM tasks WHERE id=?`, id).Scan(&rv); err != nil {
		return false, err
	}
	res, err := s.w.ExecContext(ctx, `UPDATE tasks SET status_json=?, rv=rv+1 WHERE id=? AND rv=?`, n, id, rv)
	if err != nil {
		return false, err
	}
	k, _ := res.RowsAffected()
	return k == 1, nil
}
func (s *sstore) close() { s.w.Close(); s.r.Close() }

func pct(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
	return d[int(float64(len(d)-1)*p)]
}

func main() {
	mode := flag.String("mode", "beads-embedded", "beads-embedded | beads-config | sqlite")
	dir := flag.String("dir", "", "work dir (beads-config: path to .beads)")
	writersList := flag.String("writers", "1,4,16,32", "")
	dur := flag.Duration("dur", 10*time.Second, "per phase")
	flag.Parse()
	ctx := context.Background()

	var st store
	switch *mode {
	case "beads-embedded":
		s, err := beads.Open(ctx, filepath.Join(*dir, ".beads", "embeddeddolt"))
		must(err)
		st = &bstore{s}
	case "beads-best":
		bd := filepath.Join(*dir, ".beads")
		must(os.MkdirAll(bd, 0o755))
		if _, err := os.Stat(filepath.Join(bd, "metadata.json")); err != nil {
			must(os.WriteFile(filepath.Join(bd, "metadata.json"), []byte(`{"backend":"dolt","dolt_mode":"embedded","dolt_database":"beads"}`), 0o644))
		}
		s, err := beads.OpenBestAvailable(ctx, bd)
		must(err)
		st = &bstore{s}
	case "beads-config":
		s, err := beads.OpenFromConfig(ctx, *dir)
		must(err)
		st = &bstore{s}
	case "sqlite":
		s, err := openSQLite(*dir)
		must(err)
		st = s
	}
	defer st.close()
	if is, ok := st.(*bstore); ok {
		_ = is.s.SetConfig(ctx, "issue_prefix", "bn")
	}

	var ws []int

	for _, p := range splitInts(*writersList) {
		ws = append(ws, p)
	}
	for _, w := range ws {
		ids := make([]string, w)
		for i := range ids {
			id, err := st.create(ctx, fmt.Sprintf("w%d-%d", w, i))
			must(err)
			ids[i] = id
		}
		shared, err := st.create(ctx, fmt.Sprintf("shared-%d", w))
		must(err)

		// Phase A: independent writes (each writer owns one row).
		run := func(name string, f func(i int) (bool, error)) {
			var mu sync.Mutex
			var lat []time.Duration
			var ok, conflict, errs atomic.Int64
			var firstErr atomic.Value
			deadline := time.Now().Add(*dur)
			var wg sync.WaitGroup
			for i := 0; i < w; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					var local []time.Duration
					for time.Now().Before(deadline) {
						t0 := time.Now()
						good, err := f(i)
						local = append(local, time.Since(t0))
						switch {
						case err != nil:
							errs.Add(1)
							firstErr.CompareAndSwap(nil, err.Error())
						case good:
							ok.Add(1)
						default:
							conflict.Add(1)
						}
					}
					mu.Lock()
					lat = append(lat, local...)
					mu.Unlock()
				}(i)
			}
			wg.Wait()
			fe, _ := firstErr.Load().(string)
			if len(fe) > 160 {
				fe = fe[:160]
			}
			fmt.Printf("%-15s %-12s writers=%-3d ok/s=%-8.1f conflicts/s=%-8.1f errs=%-5d p50=%-10v p99=%-10v %s\n",
				*mode, name, w, float64(ok.Load())/dur.Seconds(), float64(conflict.Load())/dur.Seconds(), errs.Load(),
				pct(lat, .5).Round(time.Microsecond), pct(lat, .99).Round(time.Microsecond), fe)
		}
		n := 0
		run("independent", func(i int) (bool, error) {
			n++
			return true, st.update(ctx, ids[i], fmt.Sprintf("n%d", n))
		})
		run("cas-shared", func(i int) (bool, error) {
			return st.cas(ctx, shared, fmt.Sprintf("c%d", i))
		})
	}
}

func splitInts(s string) []int {
	var out []int
	cur := 0
	has := false
	for _, c := range s + "," {
		if c == ',' {
			if has {
				out = append(out, cur)
			}
			cur, has = 0, false
			continue
		}
		cur = cur*10 + int(c-'0')
		has = true
	}
	return out
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}
