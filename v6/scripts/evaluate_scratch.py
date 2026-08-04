import argparse
import json
from pathlib import Path

import torch

try:
    from scripts.evaluate_ko_problems import exact_match
    from scripts.train_lora import format_record
    from scripts.train_scratch import ScratchGRU
except ModuleNotFoundError:
    from evaluate_ko_problems import exact_match
    from train_lora import format_record
    from train_scratch import ScratchGRU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ScratchGRU(len(checkpoint["vocab"]) + 1, checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    reverse = {value: key for key, value in checkpoint["vocab"].items()}
    rows = []
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        prompt = f"사용자: {item['question']}\n도우미:"
        ids = [checkpoint["vocab"].get(char, 0) for char in prompt]
        generated = list(ids)
        for _ in range(128):
            x = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            with torch.inference_mode():
                next_id = int(model(x)[0, -1].argmax())
            generated.append(next_id)
            if reverse.get(next_id, "") == "\n":
                break
        text = "".join(reverse.get(index, "") for index in generated[len(ids):]).strip()
        rows.append({**item, "prediction": text, "exact": exact_match(text, item["answer"]), "valid": bool(text)})
    result = {
        "model": "scratch-gru-char",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "count": len(rows),
        "exact_accuracy": sum(row["exact"] for row in rows) / len(rows),
        "format_validity": sum(row["valid"] for row in rows) / len(rows),
        "predictions": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("count", "exact_accuracy", "format_validity", "parameter_count")}))


if __name__ == "__main__":
    main()
