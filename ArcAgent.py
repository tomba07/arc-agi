import numpy as np

from ArcProblem import ArcProblem
from Transformations import Theory
from Observations import observe
from Theories import get_theories


class ArcAgent:
    def __init__(self):
        pass

    def _extract_simplified_examples(
        self, arc_problem
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (example.get_input_data().data(), example.get_output_data().data())
            for example in arc_problem.training_set()
        ]

    def _apply(self, theory: Theory, grid: np.ndarray, obs) -> np.ndarray:
        for fn in theory:
            grid = fn(grid, obs)
        return grid

    def _validate_theory(self, theory: Theory, obs) -> bool:
        try:
            for ex in obs.examples:
                obs.shapes = ex.input_shapes
                result = self._apply(theory, ex.input, obs)
                if not np.array_equal(result, ex.output):
                    return False
            return True
        except Exception:
            return False

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = self._extract_simplified_examples(arc_problem)
        test_input = arc_problem.test_set().get_input_data().data()

        obs = observe(examples, test_input)

        for theory in get_theories(obs):
            if self._validate_theory(theory, obs):
                print(f"{arc_problem.problem_name()}: matched")
                obs.shapes = obs.test.shapes
                return [self._apply(theory, obs.test.input, obs)]

        print(f"{arc_problem.problem_name()}: no match")
        return []
