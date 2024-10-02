from .libs import np, sym, scipy, sys
from sympy import Matrix, hessian
from .libs import List, Tuple, Dict, Optional, Callable, Any, Union

GRAD_TOL = 1e-6
MAX_ITERAND_NORM = 1e16

class IterandUpdater(object):
	def __init__(self):
		self.points: List[float] = []

	def update_points(self, points, time):
		for t, point in zip(time, points):
			self.points[t] = point

class IterativeGradient(object):
	def __init__(self, **kwargs):
		self.MAX_CORRECTIONS_REACHED = False
		self.iterand_updater = IterandUpdater()
		self.total_iterations: Optional[int] = None
		self.func_values: List[float] = []
		self.gradient_norms: List[float] = []
		self.iterands: List[float] = []
		self.backtracking_fails: List[int] = []
		self.convergence_accepted: Optional[bool] = None

	def get_experiment_scores(self):
		return  {
				"total_iterations": self.total_iterations,
				"func_values": self.func_values,
				"gradient_norms": self.gradient_norms,
				"iterands": self.iterands,
				"backtracking_fails": self.backtracking_fails,
				"convergence_accepted": self.convergence_accepted
				}

	def update_iterands(self, points, times):
		self.iterand_updater.update_points(points, times)

	def save_iterand(self, point):
		self.iterand_updater.points.append(point)

	def init_update(self):
		pass

	def update_direction(self):
		pass

	def update_step_size(self):
		pass

	def update_point(self, new_iterand):
		pass

	def current_iterand(self):
		return self.iterand_updater.points[0]

	def backtracking(self, t, point_grad, direction, t_min, alpha):
		t_b = max(t[0], t_min)
		curr_point = self.current_iterand()
	    # Backtracking until Armijo is satisfied
		grad_dot = np.dot(point_grad, direction)
		number_fails = 0
		while (True):
			is_greater = self.function(*(curr_point + t_b*direction)) > self.function(*curr_point) + alpha*t_b*grad_dot
			if (is_greater):
				t_b = t_b / 2.0
				number_fails += 1
			else:
				break
		self.backtracking_fails.append(number_fails)
		t[0] = t_b


	def iterative_gradient(
		self,
		x0: float,
		f: Union[Callable[[int], sym.core.add.Add], Callable[[Any], sym.core.add.Add]],
		max_iter: int = 150,
		second_order: Optional[bool] = False,
		backtracking: Optional[bool] = True,
		record_iterands: Optional[bool] = False,
		linear_rest: Optional[bool] = False,
		newton_args: Optional[Dict] = {},
		backtracking_args: Optional[Dict] = {}
		):
		
		'''
		x0: starting point for the problem;

		f: function to perform minimization;

		point update: function to deal with point update at each iteration;

		t: fixed step size, defaults to None;

		max_iter: maximum number of iterations;

		second_order: True for second order methods, defaults to False;

		OPTIONAL ARGUMENTS:

			newton_args:

				- hessian_correction: wether to use hessian correction in case of non PD hessian at any step.
				- correction_factor: positive constant to perform diagonal increasing in case of hessian_correction.
			;
			backtracking_args:
				
				- alpha: constant factor of Armijo condition search, must be in (0,1) interval. 
				- t_min: minimum step size to be set at the beginning of every iteration.
		'''
		fs = sorted(list(f.free_symbols), key=lambda x: x.name)
		grad_sym = Matrix([f]).jacobian(Matrix(fs))
		self.grad = sym.lambdify(fs, grad_sym, 'numpy')
		self.function = sym.lambdify(fs, f, 'numpy')

		if second_order:
			self.hess = sym.lambdify(fs, grad_sym.jacobian(Matrix(fs)), 'numpy')
			self.newton_args = newton_args

		self.use_backtracking = backtracking
		if backtracking:
			self.backtracking_args = backtracking_args

		grad_norm = float('inf')
		self.total_iterations = it = 0
		self.iter_update = IterandUpdater()
		self.save_iterand(x0)
		self.init_update()

		while True:
			curr_point = self.current_iterand()

			d = self.update_direction()
			if self.MAX_CORRECTIONS_REACHED:
				self.MAX_CORRECTIONS_REACHED=False
				self.convergence_accepted = False
				return None
			t = self.update_step_size()

	        # step to obtain the new iterand 
			x_new = curr_point + t * d
			#print("x_new", x_new)
			if np.linalg.norm(x_new, ord=np.inf) > MAX_ITERAND_NORM:
				self.convergence_accepted = False
				return None
			if linear_rest:
				grad_norm = np.linalg.norm(np.matmul(np.array(self.Z.T, dtype=float), self.grad(*x_new).squeeze().T), ord=2)
			else:
				grad_norm = np.linalg.norm(self.grad(*x_new).squeeze(), ord=2)
			
	        # iterand update
			self.update_point(x_new)
			it += 1
			self.total_iterations = it
			self.func_values.append(self.function(*curr_point))
			self.gradient_norms.append(np.linalg.norm(self.gradient_point, ord=2))
			if record_iterands:
				self.iterands.append(curr_point)

			if (grad_norm < GRAD_TOL or it >= max_iter):
				self.func_values.append(self.function(*x_new))
				self.gradient_norms.append(grad_norm)
				if record_iterands:
					self.iterands.append(x_new)
				if grad_norm < GRAD_TOL: self.convergence_accepted = True
				else: self.convergence_accepted = False

				break


