import numpy as np

from ArcProblem import ArcProblem
from helper.Indications import compute_indications
from helper.Observations import observe_example, observe_problem
from helper.Theories import generate_phase1_theories, generate_phase2_theories


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        training = arc_problem.training_set()
        examples = [
            (ts.get_input_data().data(), ts.get_output_data().data()) for ts in training
        ]
        observations = [observe_example(inp, out) for inp, out in examples]
        problem = observe_problem(observations)
        indications = compute_indications(problem)
        test_input = arc_problem.test_set().get_input_data().data()

        for theory in generate_phase1_theories(problem, indications):
            if theory.validate(examples):
                print(f"{arc_problem.problem_name()}: matched theory '{theory.name}' (phase 1)")
                return [theory.apply(test_input)]

        for theory in generate_phase2_theories(problem, indications):
            if theory.validate(examples):
                print(f"{arc_problem.problem_name()}: matched theory '{theory.name}' (phase 2)")
                return [theory.apply(test_input)]

        print(f"{arc_problem.problem_name()}: no theory matched")
        return []
