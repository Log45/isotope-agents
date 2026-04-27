## Poster Graphics and Captions

Scope used for poster metrics:
- Experiments: `exp18`, `exp19`, `exp20`, `exp24`, `exp25`
- Included rows: entries with `items_total > 0`
- Rationale: excludes near-empty-output runs so graphics represent extraction behavior rather than pipeline failures

Primary metrics:
- `micro_f1`: extraction correctness against ground truth
- `hallucination_rate`: fraction of predicted items not supported by gold
- `task_success_rate`: fraction of runs passing benchmark success criteria
- `runtime_seconds`: throughput/latency proxy

Supporting metrics (small text callout, not headline):
- `jaccard_micro`, `protocol_completeness`, `schema_validity`, `duplicate_rate`

### Figure 1: Quality by method
File: `benchmarks/poster/fig1_quality_by_method.png`

Caption:
`micro_f1` distribution across extraction methods. `FULL_SUMMARY` has the highest mean quality (0.404), `SECTION_SUMMARIES` is similar in central tendency (0.382) but with substantially higher latency, and `RAG` trails in accuracy (0.254).

### Figure 2: Hallucination by method
File: `benchmarks/poster/fig2_hallucination_by_method.png`

Caption:
Hallucination rate by method (lower is better). `FULL_SUMMARY` is best on average (0.435), while `RAG` (0.479) and `SECTION_SUMMARIES` (0.477) show higher fabrication risk.

### Figure 3: Success rate by method
File: `benchmarks/poster/fig3_success_rate_by_method.png`

Caption:
Benchmark pass/fail rate (`task_success_rate`) by method. Success is low overall under strict thresholds: `FULL_SUMMARY` 7.1%, `SECTION_SUMMARIES` 7.1%, `RAG` 0%.

### Figure 4: Speed-quality trade-off
File: `benchmarks/poster/fig4_runtime_vs_quality.png`

Caption:
Runtime vs quality (`micro_f1`) for each run. `SECTION_SUMMARIES` occupies a high-latency region (mean 384s) without commensurate quality gain over `FULL_SUMMARY` (mean 43s), which offers the best quality-efficiency balance in this scope.

### Caveat text for poster footer
`protocol_completeness` and `schema_validity` are structural metrics; they measure section/population and field presence, not semantic correctness. Main claims therefore rely on `micro_f1`, `hallucination_rate`, and `task_success_rate`.
