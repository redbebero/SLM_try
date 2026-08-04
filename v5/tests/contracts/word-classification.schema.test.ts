import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import wordClassificationSchema from "../../schemas/word-classification.schema.json" with { type: "json" };

const ajv = new Ajv({ allErrors: true, strict: true });
const validate = ajv.compile(wordClassificationSchema);

describe("word classification schema", () => {
  it("accepts multiple attributes for one Korean token", () => {
    expect(validate({
      schema_version: 1,
      token: "붉은",
      attributes: [
        { kind: "ELEMENT", value: "FIRE", delta: 1, confidence: 0.99 },
        { kind: "SIZE", value: "LARGE", delta: 0, confidence: 0.7 }
      ],
      unknown: false
    })).toBe(true);
  });

  it("accepts an unknown token with no attributes", () => {
    expect(validate({
      schema_version: 1,
      token: "평범한",
      attributes: [],
      unknown: true
    })).toBe(true);
  });

  it("rejects game-state mutation fields", () => {
    expect(validate({
      schema_version: 1,
      token: "불",
      attributes: [{ kind: "ELEMENT", value: "FIRE", delta: 1, confidence: 1 }],
      unknown: false,
      mana: 999
    })).toBe(false);
  });
});
