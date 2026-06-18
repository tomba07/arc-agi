import numpy as np

from ArcProblem import ArcProblem
from Transformations import Program, ApplyState
from Observations import observe
from Theories import synthesize


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

    def _apply(self, program: Program, source_grid: np.ndarray, source_shapes) -> np.ndarray:
        state = ApplyState(
            grid=source_grid.copy(),
            source_grid=source_grid,
            source_shapes=source_shapes,
        )
        for step in program:
            state = step(state)
        return state.grid

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = self._extract_simplified_examples(arc_problem)
        test_input = arc_problem.test_set().get_input_data().data()

        obs = observe(examples, test_input)
        programs = synthesize(obs)

        if programs:
            print(f"{arc_problem.problem_name()}: matched")
            return [self._apply(programs[0], obs.test.input, obs.test.shapes)]

        print(f"{arc_problem.problem_name()}: no match")
        return []
