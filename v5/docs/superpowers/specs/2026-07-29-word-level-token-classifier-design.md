# Word-Level Korean Incantation Classifier

## Goal

Replace the current whole-incantation proposal experiment with an independent,
smaller AI experiment: classify each whitespace-delimited Korean token into
zero or more bounded magic attributes. Godot will tokenize, accumulate, combine,
validate, and execute those attributes.

## Contract

Input to AI: one normalized Korean token only. No sentence, prefix, player
state, mana, HP, damage, inventory, or NPC state.

Output from AI:

```json
{
  "schema_version": 1,
  "token": "붉은",
  "attributes": [
    {"kind": "ELEMENT", "value": "FIRE", "delta": 1, "confidence": 0.99}
  ],
  "unknown": false
}
```

The `attributes` list supports multiple independent labels for one token. A
token with no accepted label returns `attributes: []` and `unknown: true`.
`delta` is a bounded classification output in `[-2, 2]`; it is not game state.

## Ontology

The first data-driven ontology contains:

- `ELEMENT`: FIRE, WATER, AIR, EARTH, ICE, LIGHT, SHADOW, LIFE, POISON, THUNDER, VOID
- `FORM`: ORB, SPEAR, SHIELD, BEAM, WALL, WAVE, ARROW, CHAIN, ZONE, TRAP
- `TARGET`: SELF, ALLY, ENEMY, AREA, GROUND, OBJECT, TOUCH, PROJECTILE
- `INTENT`: DAMAGE, DEFEND, CONTROL, MOVE, TRANSFORM, RESTORE, SENSE, SUMMON, DISPEL, CREATE, DESTROY, TELEPORT, ABSORB, ENCHANT
- `MODIFIER`: POWER_UP, SPEED_UP, RANGE_UP, DURATION_UP, PRECISION_UP, AREA_UP, PIERCE, SEEK, REPEAT, REFLECT, BURN, FREEZE, STUN, SILENCE
- `SIZE`: TINY, SMALL, MEDIUM, LARGE, HUGE, ALL
- `DIRECTION`: FORWARD, BACKWARD, UP, DOWN, LEFT, RIGHT, INWARD, OUTWARD, AROUND
- `QUANTITY`: ONE, FEW, MANY, ALL
- `CAST`: CHARGE, FOCUS, RELEASE, STOP, CONTINUE, NOW

Stable English labels are contract values. Korean surface forms, including
synonyms and particle-attached forms, live only in training data.

## Data strategy

`data/token-lexicon.json` is the source of truth for human-reviewed mappings.
Each entry stores surface forms, one or more attributes, and provenance. A
generator expands entries into one-token JSONL records. The generator may add
safe spacing/particle variants, but must never invent semantic labels.

Examples:

- `붉은`, `빨간`, `붉다`, `화염`, `불꽃`, `타오르는` → `ELEMENT:FIRE`
- `구체`, `공`, `둥근` → `FORM:ORB`
- `강한`, `강하게`, `세게` → `MODIFIER:POWER_UP`, `delta:1`
- `빠른`, `빠르게`, `신속히` → `MODIFIER:SPEED_UP`, `delta:1`
- `적`, `적에게`, `적을` → `TARGET:ENEMY`

Hard negatives include ordinary uses of ambiguous words, conversational words,
and unknown fantasy words. They are not assigned magic attributes.

## Model

Use the existing tokenizer-free shifted UTF-8 byte encoder with a shorter
token limit. A small pooled byte embedding model outputs:

- `attribute_logits`: one sigmoid logit per atomic `KIND:VALUE` label
- `delta_logits`: five-way delta logits for each atomic label

Training uses binary cross-entropy for attributes and cross-entropy for delta
only where an attribute is present. Inference applies a fixed threshold,
returns predicted labels, and uses the per-attribute winning delta. No rule
combination is inside the model.

## Compatibility and migration

The current `SpellProposal` model remains as a recorded failed experiment and
is not deleted. New artifacts use `token_ai.pt` and `token_ai.onnx`. The new
schema and Python modules are separate so old contract tests remain meaningful.

## Verification

- Schema rejects HP/mana/damage/game-state fields and unsupported labels.
- Lexicon generator deterministically emits all reviewed entries.
- Model supports multiple attributes and unknown tokens.
- Training input contains only token text and target attributes.
- ONNX output names/shapes are stable and match PyTorch within tolerance.
- A held-out surface form test checks synonym generalization; training accuracy
  alone is not a release criterion.
