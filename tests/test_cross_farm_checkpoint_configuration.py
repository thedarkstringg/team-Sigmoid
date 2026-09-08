from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class CrossFarmCheckpointConfigurationTests(unittest.TestCase):
    def test_run_all_sh_cross_farm_stage_uses_designated_checkpoint(self):
        content = (REPO_ROOT / "run_all.sh").read_text(encoding="utf-8")
        stage_start = content.index("Stage 5: Cross-farm generalization")
        stage_5 = content[stage_start:stage_start + 800]

        self.assertIn("checkpoints/farm_c_lr5e4/best.pt", stage_5)
        self.assertIn("--dropout 0.4", stage_5)
        self.assertNotIn("farm_c_power_residual", stage_5)
        self.assertNotIn("farm_c_final", stage_5)

    def test_cross_farm_evaluate_docstring_references_designated_checkpoint(self):
        content = (REPO_ROOT / "src" / "eval" / "cross_farm_evaluate.py").read_text(encoding="utf-8")

        self.assertIn("checkpoints/farm_c_lr5e4/best.pt", content)
        self.assertIn("10-feature", content)
        self.assertNotIn("checkpoints/farm_c_power_residual/best.pt", content)


if __name__ == "__main__":
    unittest.main()
