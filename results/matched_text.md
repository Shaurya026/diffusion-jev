# Saved source-specific accuracy

| Model | Typed | JEV | ScienceQA | TabFact | Episodes |
|---|---:|---:|---:|---:|---:|
| Gemma diffusion 5ep best | 79.10 | 77.00 | 92.50 | 82.00 | 71.63 |
| Gemma diffusion 10ep best | 78.75 | 77.00 | 93.00 | 84.50 | 72.33 |
| CLM control pilot best | 77.90 | 77.50 | 92.00 | 80.00 | 70.93 |
| CLM forward pilot best | 77.90 | 77.50 | 92.00 | 80.00 | 70.93 |
| CLM symmetric pilot best | 78.75 | 77.00 | 93.00 | 84.50 | 72.61 |
| CLM symmetric 30ep final | 80.75 | 80.00 | 93.00 | 85.00 | 72.75 |
| CLM symmetric 30ep selected best | 78.75 | 77.00 | 93.00 | 84.50 | 72.33 |
| clef | 71.10 | 66.50 | 97.00 | 87.00 | 80.34 |
| jev27vl | 58.05 | 91.00 | 95.00 | 86.00 | 76.83 |

Counts: 2,000 typed decisions; 200 each JEV/ScienceQA/TabFact; 712 few-shot episodes.

These are saved evaluations, not new reruns. No pooled overall score is reported. See [protocol and limitations](../docs/RESULTS.md).
