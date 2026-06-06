import numpy as np

from ArcProblem import ArcProblem
from helper.Theories import generate_theories


class ArcAgent:
    def __init__(self):
        pass

    def _validate_theory(self, fn, examples):
        try:
            return all(np.array_equal(fn(inp), out) for inp, out in examples)
        except Exception:
            return False

    def _validate_theories(self, theories, examples, test_input):
        for theory in theories:
            if self._validate_theory(theory, examples):
                return theory(test_input)
        return None

    def _validate_composed_theories(self, theories, examples, test_input):
        for level1 in theories:
            for level2 in theories:
                try:
                    if all(
                        np.array_equal(level2(level1(inp)), out)
                        for inp, out in examples
                    ):
                        return level2(level1(test_input))
                except Exception:
                    continue
        return None

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = [
            (example.get_input_data().data(), example.get_output_data().data())
            for example in arc_problem.training_set()
        ]
        test_input = arc_problem.test_set().get_input_data().data()

        theories = generate_theories()

        result = self._validate_theories(theories, examples, test_input)
        if result is None:
            result = self._validate_composed_theories(theories, examples, test_input)

        if result is not None:
            print(f"{arc_problem.problem_name()}: matched")
            return [result]

        print(f"{arc_problem.problem_name()}: no match")
        return []
