import unittest
from src.evaluate import evaluate

class TestEvaluation(unittest.TestCase):
    def test_accuracy_threshold(self):
        print("inside test_evaluation.py")
        metrics = evaluate("model.joblib", "data.csv", target_col="species")
        self.assertGreaterEqual(metrics["accuracy"], 0.80, "Accuracy below expected threshold")

if __name__ == "__main__":
    unittest.main()
