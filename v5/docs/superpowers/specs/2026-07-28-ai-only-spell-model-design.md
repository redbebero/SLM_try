# AI-Only Korean Spell Proposal Design

## Goal

Build only the AI portion of the fantasy spell system in v5. The output will be imported into a separate Godot project.

## Boundary

v5 owns training data, schemas, preprocessing, model training, evaluation, and ONNX export.

Godot owns input UI, gameplay, world state, mana, mental power, damage, animation, multiplayer authority, and final spell execution.

## Model behavior

The model reads a complete Korean incantation, not a hardcoded list of Korean magic words. It predicts a bounded semantic proposal:

```json
{
  "status": "PROPOSAL",
  "element": "FIRE",
  "form": "SPEAR",
  "target": "ENEMY",
  "power": 3,
  "speed": 2,
  "range": 1,
  "duration": 0,
  "confidence": 0.91
}
```

The model never decides final HP, mana, damage, position, inventory, or NPC state. Godot recalculates those values using the proposal and current state.

## Model options

The first implementation uses a small byte/character-level multi-task classifier. It avoids a separate Korean tokenizer and makes a compact ONNX handoff possible. A Korean pretrained encoder fine-tuned for this task remains a later comparison because it may generalize better but usually requires more model/tokenizer files.

## Data

Records contain the full incantation, target proposal, provenance, and split group. Data includes human-authored examples, source-attributed public-domain adaptations, controlled paraphrases, and hard negatives. Splits are made by complete incantation/template/source to prevent wording leakage.

## Validation

Tests cover schema boundaries, preprocessing determinism, dataset split leakage, model output ranges, unknown handling, and Python/ONNX output agreement.

## Handoff

The handoff package contains `spell_ai.onnx`, a versioned contract, label order, preprocessing rules, and sample input/output. It contains no Godot project files.
