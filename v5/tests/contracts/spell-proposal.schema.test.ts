import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import spellProposalSchema from "../../schemas/spell-proposal.schema.json" with { type: "json" };
import { validateSpellProposal } from "../../src/contracts/validate-contract.js";

const ajv = new Ajv({ allErrors: true, strict: true });
const validateSchema = ajv.compile(spellProposalSchema);

const validProposal = {
  schema_version: 1,
  status: "PROPOSAL",
  element: "FIRE",
  form: "SPEAR",
  target: "ENEMY",
  power: 3,
  speed: 2,
  range: 1,
  duration: 0,
  confidence: 0.91
};

describe("SpellProposal contract", () => {
  it("accepts a bounded proposal for a complete incantation", () => {
    expect(validateSpellProposal(validProposal)).toEqual({ ok: true, value: validProposal });
    expect(validateSchema(validProposal)).toBe(true);
  });

  it("accepts an unknown proposal for ambiguous input", () => {
    expect(
      validateSpellProposal({
        ...validProposal,
        status: "UNKNOWN",
        element: "UNKNOWN",
        form: "UNKNOWN",
        target: "UNKNOWN",
        power: 0,
        speed: 0,
        range: 0,
        duration: 0,
        confidence: 0.2
      })
    ).toMatchObject({ ok: true });
  });

  it("accepts LIGHT as a proposal element", () => {
    expect(validateSpellProposal({ ...validProposal, element: "LIGHT" })).toMatchObject({ ok: true });
    expect(validateSchema({ ...validProposal, element: "LIGHT" })).toBe(true);
  });

  it("rejects values outside model bounds", () => {
    expect(validateSpellProposal({ ...validProposal, power: 6 })).toMatchObject({ ok: false });
    expect(validateSpellProposal({ ...validProposal, confidence: 1.1 })).toMatchObject({ ok: false });
  });

  it("rejects game-state mutation fields", () => {
    expect(
      validateSpellProposal({
        ...validProposal,
        mana: 999,
        mental_power: 999,
        damage: 999,
        hp: 0
      })
    ).toMatchObject({ ok: false });
  });
});
