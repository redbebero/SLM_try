# v7 Korean LM

```bash
./.venv/bin/python chat.py
```

`chat.py` loads `artifacts/chat.pt` and `artifacts/tokenizer.model`, keeps
multi-turn history, and exits on `/exit` or `/quit`. Optional one-shot use:
`./.venv/bin/python chat.py --prompt '안녕하세요.'`.

`data/source` contains the copied, non-raw v6 datasets. The current chat
artifact uses a 16,384-piece tokenizer and a 50M-parameter model. A bounded
filtered KorQuAD expansion is in `data/processed`; the 21GB raw files remain
in v6 untouched.
