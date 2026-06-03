import numpy as np

from ArcProblem import ArcProblem
from helper.Observations import observe_example
from helper.Theories import SYMMETRY_THEORIES, CROP_THEORY, color_substitution_theories


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        training = arc_problem.training_set()
        examples = [
            (ts.get_input_data().data(), ts.get_output_data().data()) for ts in training
        ]
        observations = [observe_example(inp, out) for inp, out in examples]
        test_input = arc_problem.test_set().get_input_data().data()

        theories = [
            *SYMMETRY_THEORIES,
            CROP_THEORY,
            *color_substitution_theories(observations),
        ]

        for theory in theories:
            if theory.validate(examples):
                print(f"{arc_problem.problem_name()}: matched theory '{theory.name}'")
                return [theory.apply(test_input)]

        print(f"{arc_problem.problem_name()}: no theory matched")
        return []
