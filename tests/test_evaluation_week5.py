import unittest
from src.evaluate_week5 import evaluate

class TestEvaluation(unittest.TestCase):
    def test_accuracy_threshold(self):
        metrics = evaluate(data_path="data.csv", target_col="species", metric_name="accuracy", experiment_name="IRIS classifier experiment: Week5GA")
        self.assertGreaterEqual(metrics["accuracy"], 0.80, "Accuracy below expected threshold")

if __name__ == "__main__":
    unittest.main()