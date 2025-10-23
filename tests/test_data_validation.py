import unittest
from src.validate import basic_schema_checks
from src.evaluate_week5 import load_data

class TestDataValidation(unittest.TestCase):
    def test_schema_and_ranges(self):
        print("Starting data validation")
        df = load_data("data.csv")
        basic_schema_checks(df)  # raises AssertionError if invalid
        print("Data validation completed")

if __name__ == "__main__":
    unittest.main()
