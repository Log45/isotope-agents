## Results

We evaluated extraction quality against manually curated ground truth across three pipeline methods (`RAG`, `FULL_SUMMARY`, `SECTION_SUMMARIES`) using five benchmark experiments (`exp18`, `exp19`, `exp20`, `exp24`, `exp25`) with non-empty outputs.

Headline metrics:
- Quality: `micro_f1`
- Reliability: `hallucination_rate` and `task_success_rate`
- Efficiency: `runtime_seconds`

Method-level outcomes:
- `FULL_SUMMARY`: highest mean quality (`micro_f1 = 0.404`) with lower hallucination (`0.435`) and moderate runtime (`43s`)
- `RAG`: lower mean quality (`0.254`) and higher hallucination (`0.479`), with similar runtime (`42s`)
- `SECTION_SUMMARIES`: near-`FULL_SUMMARY` quality (`0.382`) but much slower (`384s`) and no hallucination advantage (`0.477`)

Interpretation:
- `FULL_SUMMARY` gives the strongest quality-efficiency trade-off in this benchmark scope.
- `SECTION_SUMMARIES` is costly in runtime and does not materially reduce hallucinations.
- `RAG` is fastest only in median terms and underperforms on correctness/reliability.

Figure placements (recommended):
1. `fig1_quality_by_method.png` (top-left)
2. `fig2_hallucination_by_method.png` (top-right)
3. `fig3_success_rate_by_method.png` (bottom-left)
4. `fig4_runtime_vs_quality.png` (bottom-right)

Limitations note:
Structural metrics (`protocol_completeness`, `schema_validity`) can overstate quality when outputs are well-formed but semantically wrong; therefore, poster claims are grounded primarily in `micro_f1`, `hallucination_rate`, and `task_success_rate`.
