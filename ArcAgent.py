import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet
from helper.MTObjectUtils import find_objects, object_to_mask, normalized_shape

class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        grid = [
            [0, 0, 2, 2, 0],
            [0, 0, 2, 2, 0],
            [1, 0, 0, 0, 3],
            [1, 0, 4, 0, 3],
            [0, 0, 4, 4, 3],
        ]

        objects = find_objects(grid)

        for obj in objects:
            print(obj)
            print("normalized:", normalized_shape(obj))
            print("mask:")
            for row in object_to_mask(obj):
                print(row)
            print()
        # predictions: list[np.ndarray] = list()

        # #Hard Coded Prediction for now
        # predictions.append(np.array([
        #     [0, 0, 6, 6, 6, 6],
        #     [0, 0, 6, 0, 0, 0],
        #     [6, 0, 6, 0, 0, 0],
        #     [6, 6, 6, 6, 0, 0]
        # ]))

        # return predictions
