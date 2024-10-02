from .IterativeGradient import IterativeGradient
from .libs import np

class CauchyGradientMethod(IterativeGradient):
    def __init__(self, **kwargs):
        super().__init__()
        self.backtracking_args = {}
        self.gradient_point = None
        self.hessian_point = None

    def update_direction(self):
        curr_point = self.current_iterand()
        g = self.grad(*curr_point).squeeze()
        self.gradient_point = g
        d = -g
        return d

    def update_step_size(self):
        curr_point = self.current_iterand()
        g = self.gradient_point
        h = self.hess(*curr_point).squeeze()

        t = (np.dot(g, g) / np.dot(np.transpose(g), np.dot(h, g)))
        t = [self.step_size]
        if self.use_backtracking:
            self.backtracking(t, g, -g, self.backtracking_args["t_min"], self.backtracking_args["alpha"])

        self.step_size = t[0]
        return self.step_size

    def update_point(self, new_iterand):
        self.update_iterands([new_iterand], [0])

    def init_update(self):
        self.step_size = 1.0