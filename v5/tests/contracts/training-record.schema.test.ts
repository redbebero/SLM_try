import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import trainingRecordSchema from "../../schemas/training-token-record.schema.json" with { type: "json" };

const ajv = new Ajv({ allErrors: true, strict: true });
const validate = ajv.compile(trainingRecordSchema);

const humanRecord = {
  id: "ko_spell_0001_step_02",
  input: {
    incantation: "붉은 불꽃을 모아 적에게 날려",
    token: "불꽃을",
    index: 1,
    prefix: ["붉은"],
    spell_context: {
      phase: "ELEMENT",
      element: null,
      form: null,
      power: 0,
      speed: 0
    }
  },
  target: {
    kind: "ELEMENT",
    value: "FIRE",
    confidence: 1
  },
  provenance: {
    type: "HUMAN_AUTHORED",
    source_id: "original_ko_0001",
    verified: true
  }
};

describe("training token record schema", () => {
  it("accepts a verified human-authored Korean incantation record", () => {
    expect(validate(humanRecord)).toBe(true);
  });

  it("requires a source URL for public-domain adaptations", () => {
    const record = {
      ...humanRecord,
      provenance: {
        type: "PUBLIC_DOMAIN_ADAPTATION",
        source_id: "oz_ chapter_14",
        verified: true
      }
    };

    expect(validate(record)).toBe(false);
    expect(validate.errors?.some((error) => error.keyword === "required")).toBe(true);
  });

  it("accepts an attributed public-domain adaptation", () => {
    const record = {
      ...humanRecord,
      provenance: {
        type: "PUBLIC_DOMAIN_ADAPTATION",
        source_id: "oz_chapter_14",
        source_url: "https://en.wikisource.org/wiki/The_Scarecrow_of_Oz/Chapter_14",
        verified: true
      }
    };

    expect(validate(record)).toBe(true);
  });

  it("rejects game-state mutation fields and unsupported labels", () => {
    const record = {
      ...humanRecord,
      mana: 999,
      target: {
        kind: "ELEMENT",
        value: "UNSUPPORTED_ELEMENT",
        confidence: 1
      }
    };

    expect(validate(record)).toBe(false);
  });

  it("rejects missing read-only spell context fields", () => {
    const record = {
      ...humanRecord,
      input: {
        ...humanRecord.input,
        spell_context: {
          phase: "ELEMENT",
          element: null,
          form: null,
          power: 0
        }
      }
    };

    expect(validate(record)).toBe(false);
  });
});
