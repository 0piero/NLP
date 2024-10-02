from .IterativeGradient import IterativeGradient
from .libs import np

class SpectralGradientMethod(IterativeGradient):
    def __init__(self, **kwargs):
        super().__init__()
        self.backtracking_args = {}
        self.step_size = 1.0
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
        last_point = self.last_iterand()
        g = self.gradient_point
        g2 = self.grad(*last_point).squeeze()

        ## Spectral step
        point_diff = curr_point - last_point
        grad_diff = g - g2
        t_div = np.dot(point_diff, grad_diff)
        # prevent numerical instability in float division
        t_div = t_div if abs(t_div) > 1e-8 else (1e-8 if t_div > 0 else -1e-8)
        t = [np.dot(point_diff, point_diff) / t_div]

        if self.use_backtracking:
            self.backtracking(t, g, -g, self.backtracking_args["t_min"], self.backtracking_args["alpha"])

        self.step_size = t[0]
        return self.step_size

    def update_point(self, new_iterand):
        curr_point = self.current_iterand()
        self.update_iterands([new_iterand, curr_point], [0, 1])

    def init_update(self):
        curr_point = self.current_iterand()
        random_point_2 = np.random.uniform(low=np.min(curr_point), high=np.max(curr_point), size=(len(curr_point),))
        self.save_iterand(random_point_2)

    def last_iterand(self):
        return self.iterand_updater.points[1]
