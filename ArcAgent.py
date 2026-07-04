import time
import numpy as np

from ArcProblem import ArcProblem
from Transformations import Theory
from Observations import Observations, ExampleObservations, initialize_observations
from Theories import TheoryDef, ALL_THEORIES


def apply_theory(theory: Theory, observations: Observations, example: ExampleObservations) -> np.ndarray:
    grid = example.input_grid.copy()
    for fn in theory:
        grid = fn(grid, observations, example)
    return grid


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
                np.array_equal(apply_theory(theory.transforms, observations, observations.example_observations[i]), out)
                for i, (_, out) in enumerate(examples)
            )
        except Exception:
            return False

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = self._extract_simplified_examples(arc_problem)
        test_input = arc_problem.test_set().get_input_data().data()
        t_start = time.perf_counter()

        observations = initialize_observations(examples, test_input)
        completed_observations: set = set()

        for theory in ALL_THEORIES:
            for check in theory.required_checks:
                if check not in completed_observations:
                    check(observations)
                    completed_observations.add(check)

            if not theory.condition(observations):
                continue

            if self._validate_theory(theory, examples, observations):
                time_spent = (time.perf_counter() - t_start) * 1000
                print(f"{arc_problem.problem_name()}: matched '{theory.name}' ({time_spent:.1f}ms)")
                return [apply_theory(theory.transforms, observations, observations.test_observations)]

        time_spent = (time.perf_counter() - t_start) * 1000
        print(f"{arc_problem.problem_name()}: no match ({time_spent:.1f}ms)")
        return []
