import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import classifiedTokenSchema from "../../schemas/classified-token.schema.json" with { type: "json" };
import spellStateSchema from "../../schemas/spell-state.schema.json" with { type: "json" };
import { validateClassifiedToken } from "../../src/contracts/validate-contract.js";

const ajv = new Ajv({ allErrors: true, strict: true });
const validateSpellState = ajv.compile(spellStateSchema);

describe("contract schemas", () => {
  it("accepts a classified token with an allowed magic label", () => {
    const token = {
      token: "불",
      index: 0,
      kind: "ELEMENT",
      value: "FIRE",
      confidence: 0.99
    };

    expect(validateClassifiedToken(token)).toEqual({ ok: true, value: token });
  });

  it("rejects model output that tries to mutate game state", () => {
    const token = {
      token: "불",
      index: 0,
      kind: "ELEMENT",
      value: "FIRE",
      confidence: 0.99,
      hp: 0,
      mana: 999,
      damage: 999
    };

    expect(validateClassifiedToken(token)).toMatchObject({ ok: false });
  });

  it("rejects an unknown label instead of expanding the engine contract implicitly", () => {
    const token = {
      token: "고대어",
      index: 0,
      kind: "ELEMENT",
      value: "UNSUPPORTED_ELEMENT",
      confidence: 0.51
    };

    expect(validateClassifiedToken(token)).toMatchObject({ ok: false });
  });

  it("accepts a spell state fixture", () => {
    expect(
      validateSpellState({
        schema_version: 1,
        phase: "ELEMENT",
        element: null,
        form: null,
        intent: null,
        power: 0,
        speed: 0,
        range: 0,
        duration: 0,
        stability: 100,
        mana_cost: 0,
        tokens: []
      })
    ).toBe(true);
  });
});
