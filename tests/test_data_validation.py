import unittest
from src.validate import basic_schema_checks
from src.evaluate_week5 import load_data

class TestDataValidation(unittest.TestCase):
    def test_schema_and_ranges(self):
        df = load_data("data.csv")
        basic_schema_checks(df)  # raises AssertionError if invalid

if __name__ == "__main__":
    unittest.main()
