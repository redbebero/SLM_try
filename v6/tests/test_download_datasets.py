import unittest


class DownloadManifestTest(unittest.TestCase):
    def test_manifest_has_required_provenance_fields(self):
        from scripts.download_datasets import DATASETS

        self.assertGreaterEqual(len(DATASETS), 2)
        for item in DATASETS:
            self.assertTrue(item["name"])
            self.assertTrue(item["revision"])
            self.assertTrue(item["license"])
            self.assertTrue(item["purpose"])
        self.assertEqual(DATASETS[0]["config"], "main")


if __name__ == "__main__":
    unittest.main()
