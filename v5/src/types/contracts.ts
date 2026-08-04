export type SpellPhase = "ELEMENT" | "FORM" | "MODIFIER" | "CAST" | "READY_TO_CAST";

export type Element = "FIRE" | "WATER" | "AIR" | "ICE" | "SHADOW";
export type SpellForm = "ORB" | "SPEAR" | "SHIELD";
export type SpellIntent = "DAMAGE" | "DEFEND" | "CONTROL" | "MOVE" | "TRANSFORM" | "RESTORE";
export type SpellEffect = "POWER_UP" | "SPEED_UP" | "RANGE_UP" | "DURATION_UP" | "FREEZE" | "MOVE" | "TRANSFORM";
export type ClassifiedKind = "ELEMENT" | "FORM" | "INTENT" | "MODIFIER" | "CAST" | "UNKNOWN";
export type ProposalStatus = "PROPOSAL" | "UNKNOWN";
export type ProposalElement = "FIRE" | "WATER" | "AIR" | "EARTH" | "ICE" | "LIGHT" | "SHADOW" | "UNKNOWN";
export type ProposalForm = "ORB" | "SPEAR" | "SHIELD" | "BEAM" | "UNKNOWN";
export type ProposalTarget = "SELF" | "ENEMY" | "AREA" | "UNKNOWN";

export type SpellProposal = {
  schema_version: 1;
  status: ProposalStatus;
  element: ProposalElement;
  form: ProposalForm;
  target: ProposalTarget;
  power: number;
  speed: number;
  range: number;
  duration: number;
  confidence: number;
};

export type TrainingProvenanceType =
  | "HUMAN_AUTHORED"
  | "PUBLIC_DOMAIN_ADAPTATION"
  | "SYNTHETIC"
  | "HARD_NEGATIVE";

export type TrainingSpellRecord = {
  id: string;
  split_group: string;
  input: { incantation: string; language: "ko" };
  target: SpellProposal;
  provenance: {
    type: TrainingProvenanceType;
    source_id: string;
    source_url?: string;
    verified: boolean;
  };
};

export type ClassifiedToken = {
  token: string;
  index: number;
  kind: ClassifiedKind;
  value?: Element | SpellForm | SpellIntent;
  effect?: SpellEffect;
  delta?: number;
  confidence: number;
};

export type WordAttributeKind = "ELEMENT" | "FORM" | "TARGET" | "INTENT" | "MODIFIER" | "SIZE" | "DIRECTION" | "QUANTITY" | "CAST";
export type WordAttribute = {
  kind: WordAttributeKind;
  value: string;
  delta: number;
  confidence: number;
};
export type WordClassification = {
  schema_version: 1;
  token: string;
  attributes: WordAttribute[];
  unknown: boolean;
};

export type SpellState = {
  schema_version: 1;
  phase: SpellPhase;
  element: Element | null;
  form: SpellForm | null;
  intent: SpellIntent | null;
  power: number;
  speed: number;
  range: number;
  duration: number;
  stability: number;
  mana_cost: number;
  tokens: string[];
};
