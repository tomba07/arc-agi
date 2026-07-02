import time
import numpy as np

from ArcProblem import ArcProblem
from Transformations import apply_theory
from Observations import initialize_observations, OBSERVATION_CHECKS
from Theories import TheoryDef, get_theories


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

    def _validate_theory(self, theory: TheoryDef, examples, observations) -> bool:
        try:
            return all(
                np.array_equal(apply_theory(theory.transforms, observations, i), out)
                for i, (_, out) in enumerate(examples)
            )
        except Exception:
            return False

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = self._extract_simplified_examples(arc_problem)
        test_input = arc_problem.test_set().get_input_data().data()
        t_start = time.perf_counter()

        observations = initialize_observations(examples, test_input)
        tested_theories: set[str] = set()

        for observation_check in OBSERVATION_CHECKS:
            observation_check(observations)
            for theory in get_theories(observations):
                if theory.name not in tested_theories:
                    tested_theories.add(theory.name)

                    if self._validate_theory(theory, examples, observations):
                        time_spent = (time.perf_counter() - t_start) * 1000
                        print(
                            f"{arc_problem.problem_name()}: matched '{theory.name}' ({time_spent:.1f}ms)"
                        )

                        return [apply_theory(theory.transforms, observations, None)]

        time_spent = (time.perf_counter() - t_start) * 1000
        print(f"{arc_problem.problem_name()}: no match ({time_spent:.1f}ms)")

        return []
