# Recall eval harness

Measures retrieval quality of the SPINE vector search (`RecallTool` →
`VectorStore`) against a golden set of `(query → expected files/symbols)`
pairs. Use it to attribute recall changes to specific tuning, rather than
guessing.

## Files

- `golden.jsonl` — the committed golden set (source of truth). One JSON
  object per line: `query`, `expected_files[]`, `expected_symbols[]`,
  `source` (`curated` | `mined`), `note`.
- `run_eval.py` — runs every golden query through `RecallTool` and reports
  `hit@{5,10,20}`, `MRR`, `miss@N` (the recall ceiling gap), and
  `set_recall@20`. Live script — needs a populated store + reachable
  embedding endpoint.
- `build_golden.py` — dev tool that derives `mined` rows from local
  `.spine/artifacts/*/*/research_log.json` file_maps. Writes
  `golden_mined.jsonl` for review; never overwrites `golden.jsonl`.

## Workflow

```bash
# 1. (Re)index after any indexing/embedding change — embeddings change shape.
spine index --wipe

# 2. Capture a labelled baseline.
python tests/recall_eval/run_eval.py --label baseline \
    --out tests/recall_eval/baseline.json

# 3. Make a change, re-index if it touched the index, re-run with a diff.
python tests/recall_eval/run_eval.py --label phase2-hybrid \
    --compare tests/recall_eval/baseline.json
```

## Reading the metrics

- **`miss@N`** is the most important number: queries whose expected items
  never appear in the top-N retrieval window. No reranker can recover
  these — it bounds everything downstream. Drive it toward zero with
  better indexing/embedding (Phase 1) and hybrid retrieval (Phase 2.1).
- **`hit@k`** tracks whether the researcher gets a usable anchor in the
  top-k it actually reads.
- **`MRR`** moves when a change improves *ordering* (reranking, embedding
  quality), not just presence.
- **`set_recall@20`** is a secondary signal for multi-target queries.

`curated` rows are sharp single/dual-target queries with known-good
answers; `mined` rows are realistic multi-file research topics. The report
breaks metrics down by source so you can see both.

## Extending the golden set

Add lines to `golden.jsonl` by hand for new sharp queries, or run
`build_golden.py` and fold good `mined` rows in. Keep `expected_files`
focused (≤6) so `hit@k` stays meaningful.
