import unittest
from src.evaluate_week5 import evaluate

class TestEvaluation(unittest.TestCase):
    def test_accuracy_threshold(self):
        print("Evaluation begin")
        metrics = evaluate(data_path="data.csv", target_col="species", metric_name="accuracy", experiment_name="IRIS classifier experiment: Week5GA")
        print(f"Metrics: {metrics}")
        self.assertGreaterEqual(metrics["accuracy"], 0.80, "Accuracy below expected threshold")
        print("Evaluation completed")

if __name__ == "__main__":
    unittest.main()
