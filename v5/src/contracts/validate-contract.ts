import Ajv, { type ErrorObject } from "ajv";
import classifiedTokenSchema from "../../schemas/classified-token.schema.json" with { type: "json" };
import spellProposalSchema from "../../schemas/spell-proposal.schema.json" with { type: "json" };
import trainingSpellRecordSchema from "../../schemas/training-spell-record.schema.json" with { type: "json" };
import wordClassificationSchema from "../../schemas/word-classification.schema.json" with { type: "json" };
import type { ClassifiedToken, SpellProposal, TrainingSpellRecord, WordClassification } from "../types/contracts.js";

const ajv = new Ajv({ allErrors: true, strict: true });
const validate = ajv.compile(classifiedTokenSchema);
const validateProposal = ajv.compile(spellProposalSchema);
const validateTrainingRecord = ajv.compile(trainingSpellRecordSchema);
const validateWordClassificationSchema = ajv.compile(wordClassificationSchema);

export type ContractValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; errors: ErrorObject[] };

export function validateClassifiedToken(input: unknown): ContractValidationResult<ClassifiedToken> {
  if (validate(input)) {
    return { ok: true, value: input as ClassifiedToken };
  }

  return { ok: false, errors: validate.errors ?? [] };
}

export function validateSpellProposal(input: unknown): ContractValidationResult<SpellProposal> {
  if (validateProposal(input)) {
    return { ok: true, value: input as SpellProposal };
  }

  return { ok: false, errors: validateProposal.errors ?? [] };
}

export function validateTrainingSpellRecord(input: unknown): ContractValidationResult<TrainingSpellRecord> {
  if (validateTrainingRecord(input)) {
    return { ok: true, value: input as TrainingSpellRecord };
  }

  return { ok: false, errors: validateTrainingRecord.errors ?? [] };
}

export function validateWordClassification(input: unknown): ContractValidationResult<WordClassification> {
  if (validateWordClassificationSchema(input)) {
    return { ok: true, value: input as WordClassification };
  }

  return { ok: false, errors: validateWordClassificationSchema.errors ?? [] };
}
