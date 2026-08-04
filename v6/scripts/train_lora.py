import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn.utils.rnn import pad_sequence


def format_record(item):
    return f"사용자: {item['question']}\n도우미: {item['solution']}\n정답: {item['answer']}"


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def collate_examples(examples, pad_token_id):
    input_ids = pad_sequence([torch.tensor(example["input_ids"]) for example in examples], batch_first=True, padding_value=pad_token_id)
    attention = pad_sequence([torch.tensor(example["attention_mask"]) for example in examples], batch_first=True, padding_value=0)
    labels = input_ids.clone()
    labels[attention == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    records = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    records = records[: args.max_examples] if args.max_examples else records
    examples = []
    for record in records:
        encoded = tokenizer(format_record(record), truncation=True, max_length=cfg.get("max_seq_length", 512))
        examples.append(encoded)

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-4))
    model.train()
    accumulation = cfg.get("gradient_accumulation_steps", 16)
    batch_size = cfg.get("batch_size", 1)
    for epoch in range(cfg.get("num_train_epochs", 3)):
        optimizer.zero_grad(set_to_none=True)
        for batch_index, start in enumerate(range(0, len(examples), batch_size)):
            batch = collate_examples(examples[start : start + batch_size], tokenizer.pad_token_id or tokenizer.eos_token_id)
            input_ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            (loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or start + batch_size >= len(examples):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        print(f"epoch={epoch + 1} loss={loss.item():.4f}")
        checkpoint = Path(args.output) / f"checkpoint-epoch-{epoch + 1}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "training_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
