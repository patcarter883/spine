# RFC 0001 §9.2.1 — Beads vs. SQLite multi-writer probe

This harness produced the throughput table in RFC 0001 §9.2.1. It runs the same two workloads against Beads (embedded Dolt) and against the SQLite store shape proposed in RFC §7.

- **independent:** W goroutines each update their own row.
- **cas-shared:** W goroutines read-then-compare-and-swap one shared row.

## Run

```sh
git clone https://github.com/gastownhall/beads ../beads-src && git -C ../beads-src checkout ce907a9
GOTOOLCHAIN=go1.27.3 CGO_ENABLED=1 go build -tags gms_pure_go -o bench .
mkdir -p runs/emb runs/sql
./bench -mode sqlite     -dir runs/sql -dur 5s -writers 1,4,16,32
./bench -mode beads-best -dir runs/emb -dur 5s -writers 1,4,16,32   # embedded Dolt via OpenBestAvailable
# server mode: official Dolt release (2.6.2 was used), server on loopback
mkdir -p runs/srv/.beads doltdata && (cd doltdata && dolt sql-server --host 127.0.0.1 --port 3317 &)
echo '{"backend":"dolt","dolt_mode":"server","dolt_server_host":"127.0.0.1","dolt_server_port":3317,"dolt_server_user":"root","dolt_database":"beads"}' > runs/srv/.beads/metadata.json
BEADS_DOLT_SERVER_PORT=3317 ./bench -mode beads-config -dir runs/srv/.beads -dur 5s -writers 1,4,16,32
grep -c "serialization failure" <dolt sql-server log>   # conflicts between writers on different issues
```

## Modes

| `-mode` | What it opens |
|---|---|
| `sqlite` | `modernc.org/sqlite`, WAL, `synchronous(FULL)`, one writer connection, `_txlock=immediate` |
| `beads-best` | writes `.beads/metadata.json` with `dolt_mode: embedded`, then calls `beads.OpenBestAvailable` |
| `beads-config` | `beads.OpenFromConfig` on an existing `.beads` dir (use for server mode) |

`beads-embedded` (`beads.Open`) is kept to show the API note in §9.2.1: despite its doc comment, it requires a Dolt server port.
