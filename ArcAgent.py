import numpy as np

from ArcProblem import ArcProblem
from helper.Observations import observe_example


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        print(f"\n=== {arc_problem.problem_name()} ===")
        for i, example in enumerate(arc_problem.training_set()):
            obs = observe_example(
                example.get_input_data().data(), example.get_output_data().data()
            )
            print(f"  example {i + 1}: {obs}")

        return []
