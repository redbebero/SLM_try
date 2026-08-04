# Dataset sources

| Source | Role | Revision/date | License | Status |
|---|---|---|---|---|
| `LGCNS/KorQuAD_2.0` | Korean document QA training data | downloaded 2026-08-04; train 83,486, validation 10,165 | Check dataset card before redistribution | downloaded |
| `beomi/KoAlpaca-RealQA` | Optional small Korean instruction slice | 2026-08-04 | CC-BY-SA-4.0; gated | not downloaded |
| `IkJun1/korean-qa-dataset` | Small Korean instruction-preservation slice | 2026-08-04; 18 rows | MIT per dataset card; verify before redistribution | downloaded |
| `openai/gsm8k` + `facebook/nllb-200-distilled-600M` | 108 translated Korean arithmetic records | 2026-08-04; seed 42 | Check both source/model licenses | translated and validated |
| `HAERAE-HUB/KMMLU` | Evaluation only | 2026-08-04 | Check dataset card | never train |
| Owned Korean problems | Main target-task data and OOD evaluation | v6; 300 rows; 50 templates | project-owned | created and validated |
| `openai/gsm8k` | English multi-step math for Korean translation | `740312add88f781978c0658806c59bc2815b9866`; main/train 7,473 rows | MIT | downloaded to `data/downloads` |
| `LGCNS/KorQuAD_2.0` | Korean document/table/list QA | `383f6a3d4efd5f238b4df7181d0af182f0ea8ff`; train 83,486 rows | CC BY-ND 2.0 KR | existing raw download reused and recorded |
| Dataset-only reasoning suite | Evaluation only | v1; test 900, OOD 300; six categories | project-owned | created and validated |

Rules:

- Never train on KMMLU, KMMLU-Pro, or owned dev/test/OOD data.
- Record row counts, filters, hashes, and exact revisions after download.
- Check source licenses before redistribution or publication.
