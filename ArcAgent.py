import numpy as np

from ArcProblem import ArcProblem
from Transformations import Theory, apply_theory
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

    def _validate_theory(self, theory: Theory, examples, obs) -> bool:
        try:
            return all(
                np.array_equal(
                    apply_theory(theory, inp, obs, i),
                    out,
                )
                for i, (inp, out) in enumerate(examples)
            )
        except Exception:
            return False

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = self._extract_simplified_examples(arc_problem)
        test_input = arc_problem.test_set().get_input_data().data()

        obs = observe(examples)

        for theory in get_theories(obs):
            if self._validate_theory(theory, examples, obs):
                print(f"{arc_problem.problem_name()}: matched")
                return [apply_theory(theory, test_input, obs, None)]

        print(f"{arc_problem.problem_name()}: no match")
        return []
