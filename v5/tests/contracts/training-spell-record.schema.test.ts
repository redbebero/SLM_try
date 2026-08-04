import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import trainingSpellRecordSchema from "../../schemas/training-spell-record.schema.json" with { type: "json" };
import { validateTrainingSpellRecord } from "../../src/contracts/validate-contract.js";

const ajv = new Ajv({ allErrors: true, strict: true });
const validateSchema = ajv.compile(trainingSpellRecordSchema);

const validRecord = {
  id: "ko_spell_0001",
  split_group: "human_0001",
  input: {
    incantation: "붉은 불꽃 창을 적에게 빠르게 날려",
    language: "ko"
  },
  target: {
    schema_version: 1,
    status: "PROPOSAL",
    element: "FIRE",
    form: "SPEAR",
    target: "ENEMY",
    power: 3,
    speed: 2,
    range: 1,
    duration: 0,
    confidence: 1
  },
  provenance: {
    type: "HUMAN_AUTHORED",
    source_id: "human_0001",
    verified: true
  }
};

describe("training spell record contract", () => {
  it("accepts a complete-incantation training record", () => {
    expect(validateTrainingSpellRecord(validRecord)).toEqual({ ok: true, value: validRecord });
    expect(validateSchema(validRecord)).toBe(true);
  });

  it("requires source URL for public-domain adaptations", () => {
    expect(
      validateTrainingSpellRecord({
        ...validRecord,
        provenance: { type: "PUBLIC_DOMAIN_ADAPTATION", source_id: "oz_0001", verified: true }
      })
    ).toMatchObject({ ok: false });

    expect(
      validateTrainingSpellRecord({
        ...validRecord,
        provenance: {
          type: "PUBLIC_DOMAIN_ADAPTATION",
          source_id: "oz_0001",
          source_url: "https://example.com/source",
          verified: true
        }
      })
    ).toMatchObject({ ok: true });
  });

  it("rejects token-level records and game-state labels", () => {
    expect(
      validateTrainingSpellRecord({
        ...validRecord,
        input: { incantation: "불", token: "불", language: "ko" }
      })
    ).toMatchObject({ ok: false });

    expect(
      validateTrainingSpellRecord({
        ...validRecord,
        target: { ...validRecord.target, mana: 999, damage: 999 }
      })
    ).toMatchObject({ ok: false });
  });
});
