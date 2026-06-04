import numpy as np

from ArcProblem import ArcProblem
from helper.Observations import observe_example, observe_problem
from helper.Theories import generate_candidates, search_compositions


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
        test_input = arc_problem.test_set().get_input_data().data()

        theories = generate_candidates(problem, examples)
        for name, apply_fn in search_compositions(theories, examples, max_depth=2):
            print(f"{arc_problem.problem_name()}: matched '{name}'")
            return [apply_fn(test_input)]

        print(f"{arc_problem.problem_name()}: no theory matched")
        return []
