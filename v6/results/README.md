# Results

## Current pilot

The pilot used only the 180 owned synthetic training examples. It is a pipeline check, not the final experiment.

| Model | Test exact accuracy | OOD exact accuracy | Format validity |
|---|---:|---:|---:|
| Qwen 0.8B base | 66.7% (20/30) | 66.7% (20/30) | 100% |
| Qwen 0.8B + owned-data LoRA | 100% (30/30) | 100% (30/30) | 100% |

The OOD result is encouraging but not yet strong evidence because the owned dataset is small and synthetic. Final claims require the mixed dataset, stronger OOD problems, and KMMLU evaluation.

## Final mixed-data 0.8B run

Training data: 360 records with the planned 50/30/15/5 mixture. Evaluation used the same deterministic evaluator.

| Model | Test exact accuracy | OOD exact accuracy | Format validity |
|---|---:|---:|---:|
| Qwen 0.8B base | 66.7% (20/30) | 66.7% (20/30) | 100% |
| Qwen 0.8B + final mixed-data LoRA | 76.7% (23/30) | 80.0% (24/30) | 100% |
| Scratch character GRU (729,620 params) | 0.0% (0/30) | 0.0% (0/30) | 100% |

Improvement gates: test +10.0 points, OOD +13.3 points, and higher OOD accuracy than scratch. All pass.

External smoke checks, 100 examples each:

| Task | Base | Final adapter |
|---|---:|---:|
| KMMLU math | 30.0% | 30.0% |
| ARC-Easy | 70.0% acc / 71.0% acc_norm | 71.0% acc / 73.0% acc_norm |

These are smoke checks with `--limit 100`, not full benchmark scores.

Full public checks with `lm_eval`:

| Task | Items | Base | Final adapter | Change |
|---|---:|---:|---:|---:|
| KMMLU Math (`acc`) | 300 | 27.33% ± 2.58 | 25.67% ± 2.53 | -1.67 points |
| ARC-Easy (`acc`) | 2,376 | 70.54% ± 0.94 | 72.01% ± 0.92 | +1.47 points |
| ARC-Easy (`acc_norm`) | 2,376 | 67.38% ± 0.96 | 70.12% ± 0.94 | +2.73 points |

Interpretation: the adapter improved the selected ARC-Easy general check modestly, but regressed on KMMLU Math. The clearest improvement remains the owned Korean problem-solving set (+10.0 test points, +13.3 OOD points). That set is small and partly synthetic, so this does not establish broad Korean intelligence or superiority over other Korean models.

## MiniCPM5-1B comparison

Compared `openbmb/MiniCPM5-1B` using the same evaluator, prompts, test set, and OOD set:

| Model | Test exact accuracy | OOD exact accuracy | Format validity |
|---|---:|---:|---:|
| Qwen3.5-0.8B base | 66.7% | 66.7% | 100% |
| Qwen3.5-0.8B + final LoRA | 76.7% | 80.0% | 100% |
| MiniCPM5-1B base (No Think mode) | 43.3% | 43.3% | 100% |

KMMLU Math full comparison: MiniCPM5-1B 28.0% ± 2.60 versus Qwen base 27.33% ± 2.58 and final adapter 25.67% ± 2.53. This suggests MiniCPM5 is stronger than this adapter on the general math check, while the Qwen adapter is much better on this specific Korean task format. MiniCPM5 used its documented No Think mode; the initial Think-mode run exceeded the 64-token cap on many examples and is discarded.
