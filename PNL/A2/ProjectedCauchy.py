from ..A1.CauchyGradientMethod import *
from ..A1.libs import sym
class ProjectedCauchy(CauchyGradientMethod):
	def __init__(self, **kwargs):
		super().__init__()
		self.A = np.ones((1, kwargs["size"]))
		self.N_space = sym.Matrix(self.A).nullspace()
		self.Z = np.array(sym.Matrix.hstack(*self.N_space), dtype=float)
		del self.N_space
		Q, R = np.linalg.qr(self.Z)
		del R
		self.proj_mtx = np.matmul(Q, Q.T)

	def update_direction(self):
		curr_point = self.current_iterand()
		g = self.grad(*curr_point).squeeze()
		self.gradient_point = g
		d = np.matmul(self.proj_mtx, -g)
		self.direction = d
		return d

	def update_step_size(self):
		curr_point = self.current_iterand()
		g = self.gradient_point
		h = self.hess(*curr_point).squeeze()

		t = (np.dot(g, g) / np.dot(np.transpose(g), np.dot(h, g)))
		t = [self.step_size]
		if self.use_backtracking:
			self.backtracking(t, g, self.direction, self.backtracking_args["t_min"], self.backtracking_args["alpha"])

		self.step_size = t[0]
		return self.step_size
