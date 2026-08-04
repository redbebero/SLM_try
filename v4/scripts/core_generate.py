"""Small, dependency-free generation path for KoJamo checkpoints."""

import re

import torch

from config import DROPOUT
from model import KoJamoNet, KoJamoTransformer


def _state_dict(checkpoint):
    return checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def _model_spec(state):
    if "emb_cho.weight" not in state or "head_type.weight" not in state:
        raise ValueError("expected a KoJamo checkpoint")
    if "pos_emb.weight" in state:
        layers = [int(m.group(1)) for key in state if (m := re.match(r"core.layers\.(\d+)\.", key))]
        return "transformer", state["emb_cho.weight"].shape[1], state["head_type.weight"].shape[1], max(layers, default=0) + 1
    layers = [int(m.group(1)) for key in state if (m := re.match(r"core\.weight_ih_l(\d+)$", key))]
    hidden = state["head_type.weight"].shape[1]
    cascade = state["head_jung.weight"].shape[1] > hidden or "decode_jung.weight" in state
    conditional = "decode_jung.weight" in state
    return ("conditional" if conditional else "cascade") if cascade else "independent", state["emb_cho.weight"].shape[1], hidden, max(layers, default=0) + 1


def load_model(checkpoint_path, vocab_sizes, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = _state_dict(checkpoint)
    variant, emb_dim, hidden_dim, layers = _model_spec(state)
    if variant == "transformer":
        model = KoJamoTransformer(
            vocab_sizes, emb_dim, hidden_dim, layers, num_heads=4,
            dropout=DROPOUT, max_seq_length=state["pos_emb.weight"].shape[0],
        )
    else:
        model = KoJamoNet(
            vocab_sizes, emb_dim, hidden_dim, layers,
            dropout=DROPOUT, cascade=variant in ("cascade", "conditional"),
            conditional_decoder=variant == "conditional",
        )
    model.load_state_dict(state)
    return model.to(device).eval(), device


def _joint_top3(cho, jung, jong, top_k=5):
    cv, ci = cho.topk(min(top_k, cho.shape[-1]))
    jv, ji = jung.topk(min(top_k, jung.shape[-1]))
    gv, gi = jong.topk(min(top_k, jong.shape[-1]))
    candidates = [(a + b + c, i, j, g) for a, i in zip(cv, ci) for b, j in zip(jv, ji) for c, g in zip(gv, gi)]
    _, i, j, g = max(candidates, key=lambda item: item[0].item())
    return i.view(1), j.view(1), g.view(1)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_chars=50, device="cpu", return_eos=False):
    sequence = tokenizer.encode(prompt).unsqueeze(0).to(device)
    initial_length = sequence.shape[1]
    ended = False
    for _ in range(max_new_chars):
        outputs = model(sequence)
        type_logits, cho, jung, jong, sym, eng, num = outputs
        token = torch.zeros(1, 1, 6, dtype=torch.long, device=device)
        kind = type_logits[:, -1].argmax(dim=-1).item()
        if kind == 0:
            if getattr(model, "conditional_decoder", False):
                _, c, v, f = model.conditional_decode_candidates(
                    model.encode_context(sequence), beam_size=3
                )[0]
            else:
                c, v, f = _joint_top3(cho[0, -1], jung[0, -1], jong[0, -1])
            token[0, 0, 0], token[0, 0, 1], token[0, 0, 2] = c, v, f
        elif kind == 1:
            token[0, 0, 3] = sym[0, -1].argmax()
        elif kind == 2:
            token[0, 0, 4] = eng[0, -1].argmax()
        else:
            token[0, 0, 5] = num[0, -1].argmax()
        sequence = torch.cat([sequence, token], dim=1)
        if kind == 1 and token[0, 0, 3].item() == tokenizer.sym_vocab.get("\n"):
            ended = True
            break
    text = tokenizer.decode(sequence[0, initial_length:]).rstrip("\n")
    return (text, ended) if return_eos else text
