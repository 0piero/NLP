from IterativeGradient import IterativeGradient
from CauchyGradientMethod import CauchyGradientMethod
from GradientStandard import GradientStandard
from NewtonMethod import NewtonMethod
from SpectralGradient import SpectralGradientMethod
from TestingFunctions import TestingFunctions
from libs import np, Type
import argparse
import random
from argparse import ArgumentParser


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            # Convert value to specified type
            if key in parser.types:
                value = parser.types[key](value)
            getattr(namespace, self.dest)[key] = value

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed value')
    parser.add_argument('--method', type=lambda x: globals().get(x))
    parser.add_argument('--func', type=lambda x: getattr(TestingFunctions, x, None))
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--second-order', type=bool, default=False)
    parser.add_argument('--fixed_step', type=float, default=1.0)
    parser.add_argument('--backtracking', type=bool, default=False)
    parser.add_argument('--newton-args', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--backtracking-args', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--problem-sizes', nargs='+', type=int)
    parser.types = {'alpha': float, 't_min': float, 'hessian_correction': bool, 'correction_factor': float}

    args = parser.parse_args()
    
    seed = args.seed
    method = args.method
    func = args.func
    max_iter  = args.max_iter
    newton_args = args.newton_args
    backtracking_args = args.backtracking_args
    problem_sizes = args.problem_sizes
    fixed_step = args.fixed_step
    second_order = args.second_order
    backtracking = args.backtracking
    random.seed(seed)
    np.random.seed(seed=seed)
    kwargs = {
        'step_size': fixed_step
    }
    args_dict = dict(
        x0 = np.random.uniform,
        f=func,
        max_iter=max_iter,
        second_order=second_order,
        backtracking=backtracking,
        newton_args=newton_args,
        backtracking_args=backtracking_args,
        SEED_NUMPY=seed
    )
    TestingFunctions.run_tests(method, args_dict, problem_sizes = problem_sizes, **kwargs)