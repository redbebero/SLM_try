# SLM v6 History

## [2026-08-03] Direction: Korean problem-solving model via fine-tuning

- Goal: build a small model that solves Korean problems well, not a general Korean language model trained from scratch.
- v4 showed that custom Jamo/GRU/Transformer models can learn Korean character structure, but pure free-form semantic generation remained weak.
- v5 showed that byte/token classifiers and constrained outputs are useful, while tiny models trained on small data do not generalize as open-ended Korean models.
- v6 therefore starts from an existing pretrained model and applies Korean-focused fine-tuning.

### Candidate models

1. `Qwen/Qwen3.5-0.8B-Base`: first pipeline and feasibility test.
2. `Qwen/Qwen3.5-2B-Base`: deferred; not part of the current v6 experiment.
3. `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`: deferred comparison candidate; check license before redistribution.

### Training direction

- Use LoRA/QLoRA, not full-parameter training.
- Prefer Korean problem-solving SFT over Korean text-only pretraining.
- Initial data mix: 70% Korean problem solving, 20% Korean instruction/dialogue, 10% capability-preservation data.
- Target tasks: arithmetic, multi-step reasoning, document QA, constraint/state reasoning, structured output, and uncertainty refusal.
- Keep train/dev/test/OOD splits separate. Do not use benchmark test data for training.

### Evaluation direction

- Use EleutherAI `lm-evaluation-harness` for KMMLU and compatible public tasks.
- Use KMMLU as a Korean knowledge/reasoning reference, not as the only success criterion.
- Add a v6-owned Korean problem-solving test set with changed wording, numbers, and problem structures.
- Compare base model vs fine-tuned model on the same prompts and decoding settings.
- Report accuracy, OOD accuracy, format validity, refusal accuracy, latency, memory, and capability regression.

### Current environment constraint

- RAM: 30 GiB; free disk observed: about 114 GB.
- `nvidia-smi` currently cannot communicate with the NVIDIA driver. GPU fine-tuning starts only after the RTX 4050 environment is restored.
- CPU-only work can validate data, tokenization, evaluation, and a small 0.8B smoke test; it is not a sensible path for long fine-tuning.

### Explicit non-goals

- No new Jamo architecture before a pretrained fine-tuning baseline.
- No claim that Korean-only text training creates general reasoning ability.
- No success claim from training loss alone.
- No full 9B fine-tuning on the current machine.

## [2026-08-03] GPU verification

- User reported that `nvidia-smi` works.
- In the current v6 shell, both `nvidia-smi` and `prime-run nvidia-smi` still report that the NVIDIA driver cannot be reached.
- System Python also has no `torch` installation.
- Do not start fine-tuning until the intended virtual environment confirms `torch.cuda.is_available() == True`.

## [2026-08-03] Refined success target

- The target is not to beat every Korean model on Korean fluency.
- The target is to take a small pretrained Qwen model, fine-tune it for Korean problem solving, and outperform comparable Korean models trained from scratch on the same Korean problem-solving evaluation.
- The comparison must include the untuned Qwen base model, the fine-tuned Qwen model, and Korean-from-scratch baselines.
- Primary score: held-out Korean problem-solving accuracy, especially changed wording, numbers, and multi-step structure.
- Secondary scores: Korean fluency, format validity, latency, memory use, and regression on general capability.

## [2026-08-03] v6 implementation plan created

- Created docs/superpowers/plans/2026-08-03-qwen-korean-problem-solving.md.
- Created the minimal README.md.
- Plan starts with Qwen3.5-0.8B QLoRA and deterministic Korean problem evaluation with KMMLU as a secondary reference.

## [2026-08-03] Scope narrowed to 0.8B and dataset acquisition

- Removed the 2B scale-up task from the current plan; larger models are deferred.
- Dataset approach: combine 300 owned Korean problem-solving seed problems and controlled variants with a limited amount of public Korean QA/instruction data.
- Use KorQuAD training data for document QA after checking its license and revision.
- Keep KMMLU and KMMLU-Pro evaluation-only to avoid benchmark contamination.
- Record dataset URLs, revisions, licenses, dates, counts, and filters in `data/SOURCES.md`.
- Add a verified Korean translation of a license-compatible English reasoning subset, beginning with GSM8K-style arithmetic; translated data supplements native Korean data and does not replace it.

## [2026-08-04] Dataset acquisition procedure explained

- Download training data through Hugging Face `datasets`.
- Use `LGCNS/KorQuAD_2.0` training data for document QA and optionally gated `beomi/KoAlpaca-RealQA` for a small Korean instruction-preservation slice.
- Keep KMMLU evaluation-only; create the target problem-solving set as owned, verified Korean problems plus controlled variants.

## [2026-08-04] Root plan created

- Created `plan.md` as the short canonical v6 plan.
- Scope is Qwen3.5-0.8B only.
- Added Korean, translated-English, KorQuAD, and evaluation-separation rules.

## [2026-08-04] v6 implementation started

- Created `pyproject.toml` and `.venv`; `uv sync` completed.
- CUDA check passed: NVIDIA GeForce RTX 4050 Laptop GPU, 5.67 GiB visible, PyTorch 2.13.0+cu130.
- Downloaded `LGCNS/KorQuAD_2.0`: 83,486 train and 10,165 validation rows.
- Generated and validated 300 owned Korean seed records: 180 train, 60 dev, 30 test, 30 OOD, with 50 template-disjoint families.
- Implemented deterministic evaluator and QLoRA trainer.
- Base baseline on owned test/OOD: 66.7% exact accuracy, 100% non-empty output.
- 20-example QLoRA smoke gate passed at 100% exact accuracy after increasing smoke-only optimization steps.
- Owned-data pilot scored 100% exact accuracy on both owned test and OOD.
- This is not final evidence; translated English data, mixed training data, stronger OOD data, and KMMLU evaluation remain.

## [2026-08-04] plan.md updated with execution status

- Marked environment, owned dataset, evaluator, baseline, and QLoRA smoke work complete.
- Added remaining gates: KorQuAD normalization, translated-English data, mixed dataset, final adapter, and KMMLU evaluation.
- Recorded the owned-data pilot as a non-final pipeline result.

## [2026-08-04] Final 0.8B experiment completed

- Normalized 1,000 KorQuAD records with streaming HTML cleanup.
- Rejected the invalid Helsinki translator output as garbage; replaced it with `facebook/nllb-200-distilled-600M` and accepted 108 Korean arithmetic translations after Korean/non-empty validation.
- Added 18 filtered Korean instruction records.
- Built `data/final`: 360 train, 60 dev, 30 test, 30 OOD; exact train mix 50/30/15/5.
- The 512-token final run OOMed; the 256-token VRAM-safe retry completed 3 epochs with loss 1.1656 → 0.9192.
- Final adapter: `models/qwen_0.8b_final_v3`.
- Final owned evaluation: base 66.7% test / 66.7% OOD; adapter 76.7% test / 80.0% OOD; format validity 100%.
- Scratch character GRU baseline: 729,620 parameters, 0% test and 0% OOD.
- External 100-example smoke: KMMLU math base 30.0%, adapter 30.0%; ARC-Easy base 70.0% acc / 71.0% acc_norm, adapter 71.0% / 73.0%.
- Primary improvement gates pass; external scores are smoke checks, not full benchmark scores.

## [2026-08-04] Full benchmark comparison completed

- Ran `lm_eval` on the full local `kmmlu_math` task: 300 items, base 27.33% ± 2.58, adapter 25.67% ± 2.53.
- Ran `lm_eval` on full `arc_easy`: 2,376 items, base 70.54% acc / 67.38% acc_norm, adapter 72.01% acc / 70.12% acc_norm.
- Interpretation: KMMLU Math regressed by 1.67 points; ARC-Easy improved by 1.47 acc points and 2.73 normalized-accuracy points.
- These are full-task results for two checks, not the full 45-subject KMMLU group. The custom Korean benchmark remains the strongest measured target-task gain, but its small synthetic design limits generalization claims.

## [2026-08-04] MiniCPM5-1B comparison

- Identified the requested model as `openbmb/MiniCPM5-1B`, a 1,080,632,832-parameter BF16 Llama architecture with 131,072-token context and Think/No-Think modes.
- Ran the same Korean evaluator in MiniCPM5's documented No Think mode: 43.3% test and 43.3% OOD, with 100% non-empty output. A default Think-mode run hit the 64-token cap and was discarded as an unfair comparison.
- Ran full KMMLU Math: MiniCPM5-1B scored 28.0% ± 2.60, compared with Qwen base 27.33% ± 2.58 and final adapter 25.67% ± 2.53.
- Result: the Qwen adapter wins decisively on this targeted Korean benchmark; MiniCPM5 is slightly better on the general math check. This is not a claim about all Korean tasks.

## [2026-08-04] Qwen3.5-27B comparison scope explained

- Verified Ollama's `qwen3.5:27b` listing: approximately 17 GB for the Q4 variant and 256K context; this machine has 5.67 GiB VRAM and only `qwen3.5:4b` is installed.
- Explained benchmark roles: owned test/OOD measure the targeted Korean task; KMMLU Math measures Korean multiple-choice math/general reasoning; ARC-Easy measures English elementary science multiple-choice; format validity only checks for non-empty output.
- A direct 27B score was not run because the checkpoint is not installed and does not fit this GPU's VRAM. It should be expected to lead on broad capability, while the specialized 0.8B adapter may still win on the narrow Korean target task.

## [2026-08-04] Plan created to beat Qwen3.5:27B on Korean reasoning

- Defined the honest target as beating a fixed 27B baseline on a narrow, verifiable Korean reasoning suite, not broad intelligence.
- Planned a new 1,200-item six-category test/OOD suite, 8,000–12,000 verified training records, 27B teacher hard-example mining, concise QLoRA SFT, optional preference training, blind comparison, and robustness gates.
- Saved the detailed plan to `docs/superpowers/plans/2026-08-04-beat-qwen27b-korean-reasoning.md`.

## [2026-08-04] Replaced 27B-teacher plan with dataset-only plan

- Removed Qwen3.5:27B from the training critical path; no 27B download is required.
- New data sources: KorQuAD 2.0, MIT-licensed GSM8K translated and verifier-checked into Korean, optional licensed OpenMath data, and a filtered Korean instruction slice.
- Added deterministic Korean reasoning generation, source/license manifests, translation verification, a 10,000-record target mixture, and a 1,200-item hidden test/OOD suite.
- Detailed plan: `docs/superpowers/plans/2026-08-04-dataset-only-qwen08b-upgrade.md`.

## [2026-08-04] Dataset-only upgrade execution started

- Built `data/reasoning/test.jsonl` with 900 items and `ood.jsonl` with 300 items across six Korean reasoning categories; IDs and templates are disjoint.
- Downloaded `openai/gsm8k` at revision `740312add88f781978c0658806c59bc2815b9866`; reused the existing KorQuAD train download at revision `383f6a3d4efd5f238b4df7181d0af182f0ea8ff`.
- Added dataset normalization and translation verification scripts. Of the existing 108 translated GSM8K rows, 89 passed and 19 were rejected.
- Generated 8,900 deterministic Korean reasoning records and assembled a validated 10,000-row train set with 60 dev, 900 test, and 300 OOD rows.
- Generated 120 robustness records.
- A full 1,200-item base generation command was too slow and left no result files; a bounded 100-item base diagnostic completed at 15.0% exact accuracy and 100% non-empty output.
- The 20-example QLoRA smoke run passed with loss 2.5094 → 2.0122. The first 500-example attempt overlapped a stale evaluator process and OOMed; stale processes were terminated. The full 10,000-row QLoRA run is currently active from a clean GPU.
- Batched training was added and tested. Batch 16 produced allocator OOM warnings; batch 12 with 192-token sequences was clean and completed a 20-example smoke run at loss 2.4249 → 1.9346.
- The full run was restarted with the safe batch-12/192-token configuration for one epoch; this is the hardware-safe execution of the dataset-only plan.
- Batched Korean evaluation support was added with `--batch-size` and `--limit`; tokenizer-level batch construction was verified for Qwen3.5-0.8B and all 15 tests still pass.

## [2026-08-04] Completed adapter made directly usable

- Confirmed the newer 10,000-row dataset-only adapter has no checkpoint yet.
- Added `scripts/chat_ko.py` for interactive Korean inference with `models/qwen_0.8b_final_v3`.
- Fixed its `BatchEncoding` generation call after a real smoke run exposed the mismatch.
- Verified the completed adapter loads and answers `한국어로 12와 7을 더하는 과정을 짧게 설명해줘` with `12 + 7 = 19`.

## [2026-08-04] Fine-tuning attempt ledger

All Qwen 0.8B fine-tuning attempts in v6:

- `models/qwen_0.8b_smoke`: initial small QLoRA smoke adapter.
- `models/qwen_0.8b_smoke_v2`: 20-example smoke adapter; exact-answer gate passed.
- `models/qwen_0.8b_owned_pilot`: trained on 180 owned Korean examples; scored 100.0% on the 30-item owned test and 30-item OOD sets. This was a pipeline check, not broad evidence.
- `models/qwen_0.8b_final_v3`: completed 360-row Korean-focused mixed-data run, 3 epochs, 256-token limit, learning rate `1e-4`. It improved the small target benchmark from 66.7% to 76.7% test and from 66.7% to 80.0% OOD. It is currently the usable adapter.
- `models/qwen_0.8b_reasoning_dataset_only_smoke20`, `_b8`, `_b12`, `_b16`, and `_batched`: trainer smoke variants used to validate batching and VRAM behavior. The batch-12 variant was clean; batch-16 produced allocator OOM warnings and was rejected.
- A 500-example dataset-only attempt overlapped a stale evaluator process and OOMed; stale GPU processes were terminated.
- A later full 10,000-row run was started and interrupted before producing a checkpoint because the original execution was too slow/non-resumable.
- A second full 10,000-row run was started with `configs/qwen_0.8b_reasoning_dataset_only.yaml`: QLoRA 4-bit, batch size 12, sequence length 192, learning rate `5e-5`, one epoch, seed 42. At the time of this entry it is actively using about 5.0 GiB of the 6.1 GiB GPU and has not yet written its epoch checkpoint.

The 10,000-row dataset-only run is intended to improve general Korean math retention while preserving the narrow Korean reasoning gain. KMMLU and ARC-Easy remain evaluation-only.

## [2026-08-04] Intermediate artifact cleanup

- Moved obsolete smoke adapters, the owned-data pilot adapter, and the scratch GRU checkpoint to `.trash/2026-08-04-intermediate/`.
- Preserved `models/qwen_0.8b_final_v3` as the currently usable model and preserved the active `models/qwen_0.8b_reasoning_dataset_only_v1` output directory.
- Preserved datasets, source manifests, benchmark results, scripts, plans, and history for reproducibility.
- Cleanup is recoverable because the environment rejected irreversible deletion commands.

## [2026-08-04] Current adapter benchmark rerun and comparison setup

- The unfinished 10,000-row training run was stopped before benchmarking because it had no checkpoint and occupied the GPU.
- A batched evaluation warning exposed right-padding on the decoder-only model. Fixed `scripts/evaluate_ko_problems.py` to left-pad and added a regression test.
- Reran the usable `models/qwen_0.8b_final_v3` with the corrected evaluator using safe batch size 1: 23/30 (76.7%) on the held-out test and 24/30 (80.0%) on OOD, with 100% non-empty output. A batched test run gave 80.0%, but batch size 1 is retained as the official score because Qwen3.5 generation changed across padding/batch paths.
- Qwen3.5-9B and Qwen2.5-27B are not installed locally. The official Qwen2.5 release lists 14B and 32B, not a 27B model; published results are therefore reported as external reference scores, not local runs.
- External reference lookup found the official Qwen3.5-9B card and official Qwen2.5 release table. The closest Qwen2.5 size to the requested 27B is Qwen2.5-32B-Instruct; it reports KMMLU 60.75 and extended MGSM8K 87.15 in the multilingual table.
