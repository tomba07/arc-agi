import numpy as np

from ArcProblem import ArcProblem
from helper.Theories import search


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = [
            (ts.get_input_data().data(), ts.get_output_data().data())
            for ts in arc_problem.training_set()
        ]
        test_input = arc_problem.test_set().get_input_data().data()

        for name, fn in search(examples):
            print(f"{arc_problem.problem_name()}: matched '{name}'")
            return [fn(test_input)]

        print(f"{arc_problem.problem_name()}: no match")
        return []
