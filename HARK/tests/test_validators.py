import unittest, sys

from HARK.validators import non_empty


class ValidatorsTests(unittest.TestCase):
    """
    Tests for validator decorators which validate function arguments
    """

    def test_non_empty(self):
        @non_empty("list_a")
        def foo(list_a, list_b):
            pass

        try:
            foo([1], [])
        except Exception:
            self.fail()

        if sys.version[0] == "2":
            with self.assertRaisesRegex(
                TypeError,
                "Expected non-empty argument for parameter list_a",
            ):
                foo([], [1])
        else:
            with self.assertRaisesRegex(
                TypeError,
                "Expected non-empty argument for parameter list_a",
            ):
                foo([], [1])

        @non_empty("list_a", "list_b")
        def foo(list_a, list_b):
            pass

        if sys.version[0] == "2":
            with self.assertRaisesRegex(
                TypeError,
                "Expected non-empty argument for parameter list_b",
            ):
                foo([1], [])
            with self.assertRaisesRegex(
                TypeError,
                "Expected non-empty argument for parameter list_a",
            ):
                foo([], [1])
        else:
            with self.assertRaisesRegex(
                TypeError,
                "Expected non-empty argument for parameter list_b",
            ):
                foo([1], [])
            with self.assertRaisesRegex(
                TypeError,
                "Expected non-empty argument for parameter list_a",
            ):
                foo([], [1])
