from .IterativeGradient import IterativeGradient
from .libs import np

class GradientStandard(IterativeGradient):
    def __init__(self, **kwargs):
        super().__init__()
        self.backtracking_args = {}
        self.step_size = kwargs.get('step_size', 1.0)
        self.gradient_point = None

    def update_direction(self):
        curr_point = self.current_iterand()
        g = self.grad(*curr_point).squeeze()
        self.gradient_point = g
        d = -g
        return d

    def update_step_size(self):
        curr_point = self.current_iterand()
        g = self.gradient_point

        t = [self.step_size]
        if self.use_backtracking:
            self.backtracking(t, g, -g, self.backtracking_args["t_min"], self.backtracking_args["alpha"])

        self.step_size = t[0]
        return self.step_size

    def update_point(self, new_iterand):
        self.update_iterands([new_iterand], [0])

    def init_update(self):
        pass

