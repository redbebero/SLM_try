"""Train the fixed-compute Korean-jamo HRM-lite on SFT-style Q/A data."""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import KoJamoDataset
from hrm_model import HRMJamoNet, HRMContextNet
from train_sft import sft_collate_fn, apply_sft_input_dropout


def masked_loss(logits, y, mask, eos_id=None, eos_weight=1.0, char_logits=None):
    total = torch.zeros((), device=y.device)
    types = ((y[:, :, 3] > 0).long()
             + 2 * (y[:, :, 4] > 0).long()
             + 3 * (y[:, :, 5] > 0).long())
    targets = (types, y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3], y[:, :, 4], y[:, :, 5])
    loss_mask = mask
    if eos_id is not None and eos_weight != 1.0:
        eos = (types == 1) & (y[:, :, 3] == eos_id)
        loss_mask = mask * torch.where(
            eos, mask.new_tensor(eos_weight), mask.new_tensor(1.0)
        )
    active_masks = (
        loss_mask,
        mask * (types == 0), mask * (types == 0), mask * (types == 0),
        loss_mask * (types == 1), mask * (types == 2), mask * (types == 3),
    )
    for prediction, target, active_mask in zip(logits, targets, active_masks):
        per_token = nn.functional.nll_loss(
            prediction.reshape(-1, prediction.size(-1)), target.reshape(-1), reduction="none"
        ).reshape_as(mask)
        total = total + (per_token * active_mask).sum() / (active_mask.sum() + 1e-8)
    loss_count = len(logits)
    if char_logits is not None:
        complete = (types == 0) & (y[:, :, 0] > 0) & (y[:, :, 1] > 0)
        char_target = ((y[:, :, 0] - 1) * 588
                       + (y[:, :, 1] - 1) * 28 + y[:, :, 2]).clamp_min(0)
        char_mask = mask * complete
        char_loss = nn.functional.nll_loss(
            char_logits.reshape(-1, char_logits.size(-1)),
            char_target.reshape(-1), reduction="none",
        ).reshape_as(mask)
        total = total + (char_loss * char_mask).sum() / (char_mask.sum() + 1e-8)
        loss_count += 1
    return total / loss_count


def first_active_mask(mask):
    """Keep only first answer position per sample for answer-start loss."""
    cumulative = mask.cumsum(dim=1)
    return mask * (cumulative == 1).to(mask.dtype)


def build_scheduled_input(x, logits, mask, probability):
    """Replace answer-side inputs with detached model predictions."""
    if probability <= 0:
        return x
    type_logits, cho_logits, jung_logits, jong_logits, sym_logits, eng_logits, num_logits = logits
    predicted_type = type_logits.argmax(dim=-1)
    predicted = torch.zeros_like(x)
    predicted[:, :, 0] = cho_logits.argmax(dim=-1)
    predicted[:, :, 1] = jung_logits.argmax(dim=-1)
    predicted[:, :, 2] = jong_logits.argmax(dim=-1)
    predicted[:, :, 3] = sym_logits.argmax(dim=-1)
    predicted[:, :, 4] = eng_logits.argmax(dim=-1)
    predicted[:, :, 5] = num_logits.argmax(dim=-1)
    predicted[:, :, 3:] = torch.where(
        (predicted_type == 1).unsqueeze(-1), predicted[:, :, 3:], torch.zeros_like(predicted[:, :, 3:])
    )
    predicted[:, :, :3] = torch.where(
        (predicted_type == 0).unsqueeze(-1), predicted[:, :, :3], torch.zeros_like(predicted[:, :, :3])
    )
    predicted[:, :, 4] = torch.where(predicted_type == 2, predicted[:, :, 4], torch.zeros_like(predicted[:, :, 4]))
    predicted[:, :, 5] = torch.where(predicted_type == 3, predicted[:, :, 5], torch.zeros_like(predicted[:, :, 5]))

    # x[t+1] is the previous prediction for y[t]. Never replace prompt tokens.
    candidate = torch.zeros_like(x)
    candidate[:, 1:] = predicted[:, :-1].detach()
    answer_input = torch.zeros_like(mask, dtype=torch.bool)
    answer_input[:, 1:] = mask[:, :-1] > 0
    replace = answer_input & (torch.rand_like(mask) < probability)
    return torch.where(replace.unsqueeze(-1), candidate, x)


def forward_segments(model, x, y, mask, segments, eos_id=None, eos_weight=1.0):
    prompt_mask = None
    answer_input = torch.zeros_like(mask, dtype=torch.bool)
    answer_input[:, 1:] = mask[:, :-1] > 0
    prompt_mask = (~answer_input) & (x.sum(dim=-1) > 0)
    state = None
    total = torch.zeros((), device=x.device)
    logits = None
    for _ in range(segments):
        logits, state = model.forward_segment(x, state, prompt_mask=prompt_mask)
        total = total + masked_loss(
            logits, y, mask, eos_id=eos_id, eos_weight=eos_weight,
            char_logits=getattr(model, "last_char_logits", None),
        ) / segments
        state = tuple(item.detach() for item in state)
    return total, logits


def build_train_val_datasets(data_dir, valid_data_dir=None, validation_ratio=0.1, seed=42):
    """Load disjoint SFT directories when a validation directory is supplied."""
    train_dataset = KoJamoDataset(data_dir=data_dir, seq_length=1000, stride=100, is_sft=True)
    if valid_data_dir:
        valid_dataset = KoJamoDataset(data_dir=valid_data_dir, seq_length=1000, stride=100, is_sft=True)
        return train_dataset, valid_dataset
    train_size = int(len(train_dataset) * (1.0 - validation_ratio))
    generator = torch.Generator().manual_seed(seed)
    return random_split(train_dataset, [train_size, len(train_dataset) - train_size], generator=generator)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = build_train_val_datasets(
        args.data_dir, args.valid_data_dir, args.validation_ratio, args.seed
    )
    eos_id = train_set.dataset.tokenizer.sym_vocab.get("\n") if hasattr(train_set, "dataset") else train_set.tokenizer.sym_vocab.get("\n")
    dataset = train_set.dataset if hasattr(train_set, "dataset") else train_set
    collate = lambda batch: sft_collate_fn(batch, max_seq_length=args.max_seq_length)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, collate_fn=collate)

    vocab_sizes = list(dataset.tokenizer.get_vocab_sizes())
    init_state = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        init_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        for index, name in enumerate(("cho", "jung", "jong", "sym", "eng", "num")):
            weight = init_state.get(f"emb_{name}.weight")
            if weight is not None and weight.ndim == 2:
                vocab_sizes[index] = weight.shape[0]

    model_cls = HRMContextNet if args.context_encoder else HRMJamoNet
    model_kwargs = dict(
        vocab_sizes=tuple(vocab_sizes), emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim, cycle_steps=args.cycle_steps,
    )
    if args.context_encoder:
        model_kwargs["context_layers"] = args.context_layers
        model_kwargs["max_seq_length"] = args.max_seq_length
        model_kwargs["use_copy"] = args.copy_head
        model_kwargs["use_current_jong"] = args.current_jong
        model_kwargs["use_joint_jamo"] = args.joint_jamo
        model_kwargs["use_query_summary"] = args.query_summary
        model_kwargs["use_char_head"] = args.char_head
    else:
        model_kwargs.update(
            use_attention=args.attention,
            use_prompt_memory=args.prompt_memory,
        )
    model = model_cls(**model_kwargs).to(device)
    if args.init_transformer:
        checkpoint = torch.load(args.init_transformer, map_location=device)
        source = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        target = model.state_dict()
        transferred = {}
        for name in ("emb_cho.weight", "emb_jung.weight", "emb_jong.weight",
                     "emb_sym.weight", "emb_eng.weight", "emb_num.weight"):
            if name in source and name in target and source[name].shape == target[name].shape:
                transferred[name] = source[name]
        if args.context_encoder:
            for name, value in source.items():
                if name.startswith("core.layers."):
                    mapped = "context_encoder." + name[len("core."):]
                    if mapped in target and target[mapped].shape == value.shape:
                        transferred[mapped] = value
            if "norm.weight" in source and "context_norm.weight" in target:
                transferred["context_norm.weight"] = source["norm.weight"]
                transferred["context_norm.bias"] = source["norm.bias"]
            if "pos_emb.weight" in source and "pos_emb.weight" in target and source["pos_emb.weight"].shape == target["pos_emb.weight"].shape:
                transferred["pos_emb.weight"] = source["pos_emb.weight"]
            # Preserve the pretrained Transformer vocabulary heads in the
            # first half of the hybrid H/L representation.
            for name in ("head_type", "head_cho", "head_jung", "head_jong",
                         "head_sym", "head_eng", "head_num"):
                weight_key = name + ".weight"
                bias_key = name + ".bias"
                if (weight_key in source and weight_key in target
                        and source[weight_key].shape[0] == target[weight_key].shape[0]
                        and source[weight_key].shape[1] * 2 == target[weight_key].shape[1]):
                    expanded = torch.zeros_like(target[weight_key])
                    expanded[:, :source[weight_key].shape[1]] = source[weight_key]
                    transferred[weight_key] = expanded
                    transferred[bias_key] = source[bias_key]
        target.update(transferred)
        model.load_state_dict(target)
        print(f"🔁 Transformer 문맥 가중치 이식: {len(transferred)}개 텐서")
    if args.init_checkpoint:
        if args.char_head or args.joint_jamo or args.query_summary:
            target_state = model.state_dict()
            compatible = {
                name: value for name, value in init_state.items()
                if name in target_state and target_state[name].shape == value.shape
            }
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            optional_prefixes = (
                "char_head.", "query_summary_proj.", "answer_start_heads.",
                "joint_", "copy_", "current_jong_proj.",
                "pos_emb.", "context_encoder.", "context_norm.", "context_skip.",
            )
            unexpected_original = set(init_state) - set(compatible)
            if (set(missing) - {name for name in target_state if name.startswith(optional_prefixes)}
                    or unexpected
                    or (unexpected_original - {name for name in init_state if name.startswith(optional_prefixes)})):
                raise RuntimeError(
                    f"Incompatible HRM checkpoint: missing={missing}, unexpected={unexpected}, "
                    f"unmatched={sorted(unexpected_original)}"
                )
        else:
            model.load_state_dict(init_state)
        print(f"🔄 HRM 초기 체크포인트 로드: {args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y, mask in tqdm(train_loader, desc=f"HRM Epoch {epoch}"):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            x = apply_sft_input_dropout(x, mask, probability=args.input_dropout)
            optimizer.zero_grad()
            teacher_loss, teacher_logits = forward_segments(
                model, x, y, mask, args.segments, eos_id, args.eos_weight
            )
            start_loss = torch.zeros((), device=x.device)
            start_logits = getattr(model, "last_answer_start_logits", None)
            if start_logits is not None and args.answer_start_weight > 0:
                start_loss = masked_loss(
                    start_logits, y, first_active_mask(mask),
                    eos_id=eos_id, eos_weight=args.eos_weight,
                )
            rollout_x = build_scheduled_input(
                x, teacher_logits, mask, args.scheduled_sampling
            )
            rollout_loss = teacher_loss
            for _ in range(max(1, args.rollout_steps)):
                rollout_loss, rollout_logits = forward_segments(
                    model, rollout_x, y, mask, args.segments, eos_id, args.eos_weight
                )
                rollout_x = build_scheduled_input(
                    rollout_x, rollout_logits, mask, args.scheduled_sampling
                )
            loss = (teacher_loss + rollout_loss) * 0.5
            loss = loss + args.answer_start_weight * start_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                state = None
                for _ in range(args.segments):
                    logits, state = model.forward_segment(x, state)
                    state = tuple(item.detach() for item in state)
                val_total += masked_loss(
                    logits, y, mask, eos_id=eos_id, eos_weight=args.eos_weight,
                    char_logits=getattr(model, "last_char_logits", None),
                ).item()
        train_loss = total / max(1, len(train_loader))
        val_loss = val_total / max(1, len(val_loader))
        print(f"HRM Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "hrm_segments": args.segments,
                "context_encoder": args.context_encoder,
                "sft_format": True,
            }, args.output)
            print(f"HRM best saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="train_data_sft_curated")
    parser.add_argument("--valid-data-dir", default=None,
                        help="Disjoint SFT validation directory; prevents validation leakage.")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/hrm_lite_best.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cycle-steps", type=int, default=4)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--scheduled-sampling", type=float, default=0.5)
    parser.add_argument("--rollout-steps", type=int, default=1,
                        help="Number of chained self-conditioned rollout passes.")
    parser.add_argument("--eos-weight", type=float, default=1.0)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--prompt-memory", action="store_true")
    parser.add_argument("--context-encoder", action="store_true",
                        help="Use a small causal context encoder before H/L HRM refinement.")
    parser.add_argument("--context-layers", type=int, default=1)
    parser.add_argument("--copy-head", action="store_true",
                        help="Add a causal pointer/copy distribution over prior input tokens.")
    parser.add_argument("--current-jong", action="store_true",
                        help="Expose the current syllable's final consonant to the context encoder.")
    parser.add_argument("--joint-jamo", action="store_true",
                        help="Use a low-rank joint score over valid (cho, jung, jong) syllables.")
    parser.add_argument("--query-summary", action="store_true",
                        help="Add a causal prefix summary before HRM output heads.")
    parser.add_argument("--char-head", action="store_true",
                        help="Add a whole Korean syllable decoder head.")
    parser.add_argument("--answer-start-weight", type=float, default=0.0,
                        help="Auxiliary loss weight for first answer token.")
    parser.add_argument("--init-transformer", default=None)
    train(parser.parse_args())
