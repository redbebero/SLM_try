import unittest


class ReasoningSuiteTest(unittest.TestCase):
    def test_builds_balanced_test_and_ood_splits(self):
        from scripts.build_reasoning_suite import build_suite

        test_rows, ood_rows = build_suite()
        self.assertEqual(len(test_rows), 900)
        self.assertEqual(len(ood_rows), 300)
        categories = {
            "arithmetic",
            "multi_step",
            "comparison",
            "state_change",
            "temporal_logic",
            "reading_inference",
        }
        self.assertEqual({row["category"] for row in test_rows}, categories)
        self.assertEqual({row["category"] for row in ood_rows}, categories)
        self.assertEqual(len({row["id"] for row in test_rows + ood_rows}), 1200)
        self.assertEqual(len({row["template_id"] for row in test_rows + ood_rows}), 1200)
        self.assertTrue(all(row["question"] and row["solution"] and row["answer"] for row in test_rows + ood_rows))


if __name__ == "__main__":
    unittest.main()
