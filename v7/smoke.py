import torch
from src.korean_lm.model import Config, KoreanLM

torch.manual_seed(0)
cfg = Config(vocab_size=256, block_size=16, n_layer=2, n_head=4, n_embd=64)
model = KoreanLM(cfg)
ids = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
with torch.no_grad():
    before = model(ids, ids)[1].item()
for _ in range(20):
    opt.zero_grad(set_to_none=True)
    loss = model(ids, ids)[1]
    loss.backward(); opt.step()
with torch.no_grad():
    after = model(ids, ids)[1].item()
assert after < before, (before, after)
print({"before": before, "after": after, "passed": True})
