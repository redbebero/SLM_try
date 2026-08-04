import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

type IncantationFixture = {
  id: string;
  source_url: string;
  license_note: string;
  source_text: string;
  korean_test_input: string;
  expected_intent: string;
};

describe("public-domain incantation fixtures", () => {
  it("keeps source attribution and no word-to-effect dictionary", async () => {
    const content = await readFile("data/test-incantations.json", "utf8");
    const fixtures = JSON.parse(content) as IncantationFixture[];

    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const fixture of fixtures) {
      expect(fixture.source_url).toMatch(/^https:\/\//);
      expect(fixture.license_note.length).toBeGreaterThan(0);
      expect(fixture.source_text.length).toBeGreaterThan(0);
      expect(fixture.korean_test_input.length).toBeGreaterThan(0);
      expect(fixture.expected_intent).toMatch(/^(CONTROL|MOVE|TRANSFORM)$/);
    }

    expect(content).not.toContain('"word"');
    expect(content).not.toContain('"mapping"');
  });
});
