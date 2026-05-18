# Chain A Evaluation Metrics — Methodology Notes

Companion document to `compute_metrics.py`. This file records the design decisions made when implementing the evaluation pipeline against the metrics recommended by Hendro. It is intended to be cited from the thesis methodology chapter and referenced if any of the metric implementations need to be audited later.

---

## Scope

The `compute_metrics.py` script computes, for each (model, excerpt) pair in the Chain A experiment, a complete set of structural graph-comparison metrics between the predicted adjacency matrix and the ground-truth adjacency matrix. The full set of metrics covers everything in Hendro's recommendation email — core metrics (adjacency P/R/F1, orientation P/R/F1, SHD), recommended additional metrics (MCC, FDR, SID), and optional descriptive metrics (accuracy, balanced accuracy, exact match ratio, skeleton-based metrics). Two ontology canonicalisations are computed in parallel: a 13-node view using the ontology as-is, and a 12-node view in which PBV and Market Value of Equity are merged into a single compound node per the canonicalisation policy in the methodology chapter.

Per Hendro's email, metrics are computed at two aggregation levels: one row per (model, excerpt) pair in `metrics_per_pair.csv`, and one row per model (averaged across all 5 excerpts) in `metrics_per_model.csv`.

---

## Design decisions

### 1. Source of truth: JSONL, not CSV

The metric script reads the original Kaggle JSONL files and rebuilds adjacency matrices in-memory, rather than parsing the Option-B CSVs produced by `build_adjacency_matrices.py`. Two reasons. First, the CSVs are a presentation format for human inspection and include inline comments (excerpt_id / firm / status markers, blank separator rows) that would need a custom parser. Second, reading from JSONL keeps the metric script independent of the CSV builder — a bug in one cannot propagate to the other. The CSVs and the metrics both derive from the same upstream JSONL data and will always stay in sync.

The matrix-construction logic in `compute_metrics.py` is a faithful copy of the logic in `build_adjacency_matrices.py`: same compound-MBV canonicalisation rules, same handling of `not_in_ontology` predictions (dropped, not counted as matrix cells), same self-loop / duplicate filtering.

### 2. Two SHD variants reported

The Structural Hamming Distance has two defensible definitions in the causal-discovery literature, and the choice affects how edge reversals are penalised. Both are reported because they measure slightly different things.

**`shd_strict`** follows Tsamardinos, Brown & Aliferis (2006) and is the variant most commonly used in LLM causal-extraction benchmarks such as ReCITE and CausalBench. It counts every cell-level disagreement in the directed adjacency matrix. A reversed edge (predicted `A → B` where ground truth says `B → A`) contributes 2 to the SHD: the cell `(A, B)` disagrees because it is 1 in prediction and 0 in ground truth, and the cell `(B, A)` disagrees because it is 0 in prediction and 1 in ground truth. Strict SHD therefore punishes reversals harder than missing or spurious edges. This is the convention cited when "SHD is sensitivity-biased" — reversals double-count, making the metric aggressive at penalising orientation errors.

**`shd_reversal`** follows the convention used by some causal-discovery packages (e.g. R's `pcalg::shd`) and the wording in Hendro's email ("additions, deletions, and reversals"). A reversal counts as 1 operation, not 2. This is implemented by computing the skeleton-level disagreement (missing and spurious skeleton edges) plus the count of shared skeleton edges whose direction differs between prediction and ground truth.

**Reporting convention.** The thesis can cite either; they answer different questions. The strict variant is better when comparing against external benchmarks that use the Tsamardinos definition. The reversal variant is better when discussing the interpretation Hendro writes ("the number of edge additions, deletions, and reversals needed to transform the predicted DAG into the ground-truth DAG") since "reversal" is treated as a primitive operation rather than a composite of deletion + addition.

### 3. Two orientation conventions reported

Hendro's email isolates orientation as a distinct dimension of evaluation, with the rationale that "a graph may recover the right connections but still assign wrong causal directions." Two conventions for computing orientation P/R/F1 exist in the literature; the script reports both.

**`ori_*` — Convention A, conditional on shared skeleton.** Orientation metrics are computed only over the set of skeleton edges present in *both* the predicted and ground-truth graphs. For each shared skeleton edge, the directed edges present in each graph are compared; correctly-directed edges count as TP, edges present in the prediction's direction but not the ground truth's count as FP, and vice versa for FN. This convention directly implements Hendro's framing: it isolates the orientation problem from the skeleton problem. A model that gets the skeleton entirely wrong but happens to correctly orient its one shared skeleton edge will still score 1.0 on orientation metrics under this convention — which is the desired behaviour for a diagnostic that answers "when the model finds the right connections, does it get the direction right?"

**`ori_*_full` — Convention B, unconditional.** Orientation metrics are computed on the full directed-edge sets without conditioning. A reversal is a simultaneous FP (the wrong direction was predicted) and FN (the correct direction was missed), which makes orientation P/R penalise reversals twice — once for the spurious wrong-direction edge and once for the missing correct-direction edge. This is more stringent and is the convention used in some LLM benchmarks including arXiv:2404.06349 (CausalBench).

**Reporting convention.** The thesis reports both; Convention A is the one that matches Hendro's verbal description, and Convention B is useful as a stricter complement when the thesis wants to claim a model "got orientation right" in a stronger sense. Under Convention A, a reversal inside the shared skeleton is penalised normally (as FP in the wrong direction + FN in the correct direction within that skeleton edge), so Convention A is not permissive — it is conditional.

### 4. TN counts depend on the node-universe choice

The script reports TP/FP/FN/TN at the cell level of the directed adjacency matrix, excluding the diagonal (self-loops are not legal edges). For the 13-node view, the non-diagonal cell count is 13 × 13 − 13 = 156, of which 3 cells are ground-truth edges, leaving 153 true negatives at a maximum (when the model predicts no edges at all). For the 12-node view the non-diagonal cell count is 132, with 3 GT edges and 129 max TN.

This is the main reason the script reports both views in parallel. TN-dependent metrics — specificity, balanced accuracy, MCC, FPR — are sensitive to how the "possible negative edge" space is defined. The 13-node view carries more TN cells because PBV and MVE are separate, which inflates specificity and balanced accuracy slightly relative to the 12-node view. The 12-node view is the one graded under the canonicalisation policy for exact-match purposes.

Specificity, accuracy, and balanced accuracy values will be close to 1.0 for every model because the ground truth is sparse (3 edges out of 132–156 possible) — Hendro's email flags this explicitly ("for sparse graphs [accuracy] may be misleading because the large number of zero entries can inflate the score"). The script reports accuracy anyway because Hendro lists it, but MCC and balanced accuracy are the more informative imbalance-robust equivalents.

### 5. SID implemented from scratch using the Peters & Bühlmann 2015 criterion

Structural Intervention Distance (SID) is the one metric in Hendro's list that has no standard Python implementation — the reference implementation is in the R package `pcalg::structIntervDist`, and the Python wrappers available (via `cdt`, the Causal Discovery Toolbox) require R to be installed. Rather than introducing an R dependency into the thesis pipeline, the script implements SID from scratch following the original definition in Peters & Bühlmann (2015), "Structural Intervention Distance for Evaluating Causal Graphs," *Neural Computation* 27(3).

The Peters & Bühlmann definition: for every ordered pair (x, y) with x ≠ y, SID counts the pair as "wrong" if the parent-adjustment set implied by the predicted DAG — that is, `Z = pa_{G'}(x) \ {y}`, the parents of x in the predicted graph excluding y — is not a valid adjustment set for the true causal effect of x on y in the ground-truth DAG. A set Z is a valid adjustment set for the effect of x on y in the true DAG G if (a) Z contains no descendants of x in G, and (b) Z blocks every non-causal path from x to y in G.

The implementation computes these two conditions directly for each (x, y) pair. Condition (a) is a simple descendant-set membership test against the ground-truth DAG. Condition (b) is equivalent to d-separation of x from y given Z in the manipulated ground-truth DAG where all incoming edges to x have been removed (the standard "do-operation" framing). The d-separation test is implemented via the canonical ancestral-moralisation algorithm — restrict to ancestors of {x, y} ∪ Z, moralise (connect all pairs of unmarried parents, then undirect), remove nodes in Z, and test graph-connectivity between x and y in the resulting undirected graph.

SID is only defined for DAGs (acyclic predicted graphs). If a model's predicted graph contains a cycle, SID is reported as blank and the `is_dag` flag in the output is 0 for that row. The `sid_computable_fraction` in the per-model aggregate records the fraction of that model's excerpts for which SID was computable. In the current pilot data, all 25 (model, excerpt) pairs produced DAGs, so SID was computable for everything.

The implementation was spot-checked against hand-derived values on simple test cases (including the trivial case of pred = GT, where SID = 0, and the case of pred = empty graph, where every (x, y) pair fails the validity check and SID equals n(n-1)). A reviewer who wants to audit SID can use these invariants plus the Peters & Bühlmann paper's worked examples.

### 6. Macro aggregation at the model level

Model-level metrics (in `metrics_per_model.csv`) are computed as the macro mean across the 5 per-excerpt values. Macro averaging means: compute F1 (or any metric) separately on each excerpt, then take the simple mean of the 5 values. This is the more common convention in NLP benchmarking and makes per-excerpt confidence intervals computable directly from the per-pair rows.

The alternative — micro averaging — would pool TP, FP, FN across all 5 excerpts and compute F1 from the pooled counts. In this experiment macro and micro would give near-identical results because the ground truth has the same edge count (3) across all excerpts, so the excerpts contribute equally to any pooled count. If future work runs Chain A on excerpts with varying GT edge counts, the choice between macro and micro becomes substantive and should be revisited.

One consequence of macro-averaging is that metrics reported as "model-level F1" in this study are the mean of 5 excerpt-level F1 scores, not a single F1 computed over pooled predictions. The per-pair CSV carries all the underlying data if micro averaging is needed for any specific comparison.

### 7. Handling malformed, none, and error records

When a model produces a record with status `malformed`, `none`, or `error`, the predicted adjacency matrix is constructed as an all-zero matrix — the model produced no valid edges to place. This gives TP = 0, FP = 0, FN = |GT edges|, TN = (max TN count). Precision becomes undefined (0/0) and is set to 0 by convention. Recall is 0 by construction. F1 is 0. MCC has a zero factor in its denominator and is also set to 0 by convention. SHD strict and SHD reversal both equal the number of GT edges (3, just deletions).

This is the defensible choice: a model that produces no edges is scored as maximally-cautious, not as "no prediction." The alternative — excluding malformed records from the aggregate — would be misleading because it would let a model that fails to converge on some excerpts appear stronger than one that converges but produces wrong predictions on the same excerpts. The status field is preserved in the per-pair CSV so any analysis can filter or stratify by it.

Qwen 3.5 9B in thinking mode is the one model in the current pilot that has status=malformed rows (3 of 5 excerpts). Its macro-averaged metrics therefore include three rows of zeros from the malformed cases plus two rows of actual predictions. Any comparison that wants to evaluate only the cases where Qwen thinking produced output should filter to `status=ok` in the per-pair CSV before aggregating — this is not done automatically because doing so would hide the convergence problem.

### 8. Normalised SHD

Hendro's recommendation list includes "Normalized SHD — SHD divided by the number of possible edges or by the number of true edges." The script reports both normalisations for each SHD variant:

- `shd_*_norm_edges`: SHD divided by the number of ground-truth edges (3 for all Chain A excerpts). Useful for comparing across graphs of different true-edge counts, which is not applicable here (all excerpts have 3 GT edges) but would matter if Chain A were extended to include partial-chain excerpts later.
- `shd_*_norm_possible`: SHD divided by the total number of possible directed edges in the node universe (132 for 12-node, 156 for 13-node). This is the sparsity-corrected normalisation.

The two normalisations are not interchangeable. The edges-normalised version is closer to an "error rate" (how many operations are needed as a fraction of the true structure), while the possible-normalised version is closer to a "disagreement density" (what fraction of all cells disagree).

### 9. Exact match ratio

Reported as a 0/1 indicator per (model, excerpt). At the model level, the mean of the 0/1 indicators equals the fraction of excerpts on which the model's predicted edge set exactly matches the ground-truth edge set. In the current pilot data this is 0 for every model — no model reconstructed the full 3-edge Chain A on any excerpt, under either the 13-node or 12-node view.

### 10. Skeleton metrics

Reported separately as `skeleton_precision`, `skeleton_recall`, `skeleton_f1`, and `skeleton_shd`. These metrics treat both the predicted and ground-truth graphs as undirected and measure whether the model identified the right *connections* regardless of their direction. They are the Convention-A-adjacent complement to orientation metrics: adjacency + skeleton are conceptually close but differ in what they count.

In this script, the adjacency metrics (`adj_*`) and skeleton metrics (`skeleton_*`) produce identical numerical values on any given (pred, GT) pair because the underlying sets are identical (unordered pairs of nodes connected by at least one directed edge in either direction). They are retained as separate columns in the output because Hendro's email lists adjacency as a core metric and skeleton as an optional descriptive metric — keeping them separate preserves the naming Hendro used and allows a thesis to cite them independently. If future work introduces multi-edge graphs (where a node pair could have multiple parallel edges), the two metrics would diverge and the separate columns would carry distinct information.

---

## Metric map: Hendro's recommendation list to output columns

| Hendro's item | Output columns | Group |
|---|---|---|
| Adjacency P/R/F1 | `adj_precision`, `adj_recall`, `adj_f1` | Core |
| Orientation P/R/F1 | `ori_precision`, `ori_recall`, `ori_f1` (conditional); `ori_precision_full`, `ori_recall_full`, `ori_f1_full` (unconditional) | Core |
| SHD | `shd_strict`, `shd_reversal` | Core |
| TP / FP / FN / TN | `tp`, `fp`, `fn`, `tn` (directed); `tp_skeleton`, `fp_skeleton`, `fn_skeleton`, `tn_skeleton` (skeleton) | Descriptive |
| Specificity / TNR | `specificity` | Descriptive |
| FDR | `fdr` | Descriptive |
| FPR / FNR | `fpr`, `fnr` | Descriptive |
| Accuracy | `accuracy` | Descriptive |
| Balanced accuracy | `balanced_accuracy` | Descriptive |
| MCC | `mcc` | Recommended additional |
| Normalised SHD | `shd_strict_norm_edges`, `shd_strict_norm_possible`, `shd_reversal_norm_edges`, `shd_reversal_norm_possible` | Descriptive |
| Exact match ratio | `exact_match` | Descriptive |
| SID | `sid`, `is_dag`, `sid_computable_fraction` | Recommended additional |
| Skeleton SHD | `skeleton_shd` | Descriptive |
| Skeleton F1 | `skeleton_f1`, `skeleton_precision`, `skeleton_recall` | Descriptive |

Every metric in Hendro's email is computed and emitted. Nothing was dropped, substituted, or renamed without documentation above.

---

## References

- Peters, J., & Bühlmann, P. (2015). Structural Intervention Distance for Evaluating Causal Graphs. *Neural Computation*, 27(3), 771–799. https://doi.org/10.1162/NECO_a_00708
- Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure learning algorithm. *Machine Learning*, 65(1), 31–78. https://doi.org/10.1007/s10994-006-6889-7
- Zhou, Y., et al. (2024). CausalBench: A comprehensive benchmark for causal learning capability of large language models. *arXiv preprint*. https://arxiv.org/abs/2404.06349
- Klar, et al. (2026). Can large language models infer causal relationships from real-world text? *arXiv preprint* (ReCITE benchmark). https://arxiv.org/abs/2505.18931
