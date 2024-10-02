from ..A1.NewtonMethod import *
from ..A1.libs import sym

class NewtonReduzido(NewtonMethod):
	def __init__(self, **kwargs):
		super().__init__()
		self.A = np.ones((1, kwargs["size"]))
		self.N_space = sym.Matrix(self.A).nullspace()
		self.Z = np.array(sym.Matrix.hstack(*self.N_space), dtype=float)
		del self.N_space

	def update_direction(self):
		curr_point = self.current_iterand()
		self.gradient_point = g = self.grad(*curr_point).squeeze()
		h = np.matmul(np.matmul(self.Z.T, self.hess(*curr_point)), self.Z)
		
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

		s = scipy.linalg.solve_triangular(L, -np.matmul(self.Z.T, g.transpose()), lower=True)
		d = scipy.linalg.solve_triangular(np.transpose(L), s, lower=False)
		self.curr_direction = np.matmul(self.Z, d)
		return self.curr_direction