import unittest
from src.validate import basic_schema_checks
from src.evaluate import load_data

class TestDataValidation(unittest.TestCase):
    def test_schema_and_ranges(self):
        print("inside test_data_validation.py")
        df = load_data("data.csv")
        basic_schema_checks(df)  # raises AssertionError if invalid

if __name__ == "__main__":
    unittest.main()
