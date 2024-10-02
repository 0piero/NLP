from .IterativeGradient import IterativeGradient
from .libs import np, scipy, List, sys
MAX_NUMBER_CORRECTIONS = 3000 

class NewtonMethod(IterativeGradient):
    def __init__(self, **kwargs):
        super().__init__()
        self.newton_args = {}
        self.hess_corrections: List[int] = []
        self.gradient_point = None
        self.curr_direction = None

    def get_experiment_scores(self):
        scores = super().get_experiment_scores()
        scores['hess_corrections'] = self.hess_corrections
        return scores


    def update_direction(self):
        curr_point = self.current_iterand()
        self.gradient_point = g = self.grad(*curr_point).squeeze()
        h = self.hess(*curr_point)

        ## Checking if hessian matrix is PD using Cholesky factorization
        is_PD = False
        number_corrections = 0
        correction_factor = self.newton_args["correction_factor"]
        while not is_PD:

            if number_corrections > MAX_NUMBER_CORRECTIONS:
                print(f"MAX_NUMBER_CORRECTIONS: {number_corrections}")
                self.MAX_CORRECTIONS_REACHED = True
                break
            try:
                L = np.linalg.cholesky(h)
                is_PD = True
            except np.linalg.LinAlgError as e:
                number_corrections += 1
                ## Hessian correction
                if self.newton_args["hessian_correction"]:
                    np.fill_diagonal(h, np.diagonal(h) + np.full(h.shape[0], correction_factor))
                    correction_factor = self.newton_args["correction_factor"]*(number_corrections**3)
                else:
                    print(e)
                    sys.exit()

        self.hess_corrections.append(number_corrections)
        if self.MAX_CORRECTIONS_REACHED:
            return None

        s = scipy.linalg.solve_triangular(L, -g.transpose(), lower=True)
        d = scipy.linalg.solve_triangular(np.transpose(L), s, lower=False)
        self.curr_direction = d
        return d

    def update_step_size(self):
        t = [1.0]
        g = self.gradient_point

        if self.use_backtracking:
            self.backtracking(t, g, self.curr_direction, self.backtracking_args["t_min"], self.backtracking_args["alpha"])
        
        return t[0]

    def update_point(self, new_iterand):
        self.update_iterands([new_iterand], [0])

    def init_update(self):
        pass