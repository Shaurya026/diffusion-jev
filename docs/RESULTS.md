# Results and comparison protocol

Scores here come from saved local evaluations, not model-card leaderboards or a new rerun.
This release reports source-specific scores only; no pooled overall metric is reported.

## Retained evaluation sources

| Source | Decisions | Scope |
|---|---:|---|
| Typed decisions | 2,000 | Complete original public test split |
| JEV | 200 | Fixed text subset |
| ScienceQA | 200 | Fixed text-only subset, not image evaluation |
| TabFact | 200 | Fixed table-text subset |
| Few-shot episodes | 712 | Repeated settings for 215 independent queries |

[The complete table](../results/matched_text.md) includes Gemma, CLEF, JEV and contrastive-loss
ablations. [The CSV](../results/matched_text.csv) preserves per-source accuracy, soft CE,
summed multiclass Brier and 15-bin ECE. Accuracy uses target argmax, with the first option
breaking ties. Earlier experiments sometimes used separately annotated labels. The CSV
records label disagreements; metric-definition differences are not model improvements.

## Comparator identity and fairness

JEV is **AutoTrust JEV-27B-VL System 1**, evaluated through its released typed-decision API.
CLEF uses its native joint-schema interface. Each retained source uses matched evidence,
demonstrations, questions and ordered options; native serialization and tokenization differ.
CLEF internally sorts choice options and probabilities are mapped back to original keys.
JEV retains its released LoRA, decision biases and shipped per-kind temperatures. Reported
proper scores use native probabilities.

These are specialist comparisons, not equal-compute or equal-training-data experiments.
CLEF leads on the few-shot episodes and several other subsets. Student training included
related task families. Historical checkpoints are not claimed to have trained exclusively
on the public recipe included here.

## Historical reproducibility issue

Repeated evaluations of identical selected Gemma adapter bytes changed predictions; the
cause remains unresolved. The five-epoch versus ten-epoch rows are descriptive saved results,
not established benefit from more training. Small differences are not proven gains. A
controlled rerun needs matched software, attention backend, batching and seeds, plus
query-level uncertainty intervals.

The demo uses the five-epoch validation-selected adapter, with its own hash, hardware and
complete record set. This does not resolve the historical issue.

## Other checkpoints and ablations

[Historical source scores](../results/historical_cohorts.csv) retain available Laya,
ModernBERT and Scout measurements under their original protocols. Some external-model
entries were not evaluated when that snapshot was written; blanks are not zeros. Those
protocols are not interchangeable with the matched text evaluation above.

Contrastive-loss trials are reported per source. The 30-epoch final endpoint and validation-
selected checkpoint are distinct; selection retained the parent. Choosing the final endpoint
because a test score is higher would be test-based selection. Architecture and loss changed
together, so the ablation does not isolate the loss effect.

## Typed-decision demonstrations

The handwritten GPU suite contains 16 predictions per model: eight cases at one and four
passes. ModernBERT matched 13, Scout 12 and Gemma 16 references. All results remain in
`assets/demo_records.json`. This is an illustration, not a benchmark. GIF playback duration
is unrelated to inference speed; first-call latency may include warmup.
