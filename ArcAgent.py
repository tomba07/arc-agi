import numpy as np

from ArcProblem import ArcProblem
from helper.Theories import generate_theories


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = [
            (example.get_input_data().data(), example.get_output_data().data())
            for example in arc_problem.training_set()
        ]
        test_input = arc_problem.test_set().get_input_data().data()

        def validates(fn):
            try:
                return all(np.array_equal(fn(inp), out) for inp, out in examples)
            except Exception:
                return False

        theories = generate_theories()

        for fn in theories:
            if validates(fn):
                print(f"{arc_problem.problem_name()}: matched")
                return [fn(test_input)]

        for f1 in theories:
            for f2 in theories:
                if validates(lambda g, a=f1, b=f2: b(a(g))):
                    print(f"{arc_problem.problem_name()}: matched")
                    return [f2(f1(test_input))]

        print(f"{arc_problem.problem_name()}: no match")
        return []
