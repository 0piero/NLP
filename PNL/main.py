from .A1.IterativeGradient import IterativeGradient
from .A1.CauchyGradientMethod import CauchyGradientMethod
from .A1.GradientStandard import GradientStandard
from .A1.NewtonMethod import NewtonMethod
from .A1.SpectralGradient import SpectralGradientMethod
from .A1.TestingFunctions import TestingFunctions
from .A1.libs import np, Type

from .A2.ProjectedCauchy import ProjectedCauchy
from .A2.NewtonReduzido import NewtonReduzido


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
#                print(type(value))
#                print(parser.types[key])
                value = parser.types[key](eval(value))
            getattr(namespace, self.dest)[key] = value

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='seed aleatoria para controle do ponto inicial')
    parser.add_argument('--method', type=lambda x: globals().get(x), help='nome do metodo que se deseja rodar, é o mesmo nome do arquivo .py que o implementa (e.g. NewtonMethod)')
    parser.add_argument('--linear_rest', type=bool, default=False, help='booleano para os problemas com restrições lineares')
    parser.add_argument('--func', type=lambda x: getattr(TestingFunctions, x, None), help='nome da função escolhida para teste (e.g. func_A)')
    parser.add_argument('--max-iter', type=int, default=100, help='numero maximo de iteracoes')
    parser.add_argument('--second-order', type=bool, default=False, help='booleano para indicar se o metodo utiliza informacao de segunda ordem')
    parser.add_argument('--fixed_step', type=float, default=1.0, help='valor para ser fornecido quando o metodo utilizar passo fixo')
    parser.add_argument('--backtracking', type=bool, default=False, help='booleano para indicar se deve ser feita busca linear para satisfazer Armijo')
    parser.add_argument('--newton-args', nargs='*', action=ParseKwargs, default={}, help='quando utilizado, deve fornecer se será feita correção de hessiana e um fator constante que será utilizado para correção a cada iteração (e.g. --newton-args hessian_correction=True correction_factor=10.0)')
    parser.add_argument('--backtracking-args', nargs='*', action=ParseKwargs, default={}, help='quando o argumento backtracking for habilitado, deve fornecer os parametros a serem utilizados na busca linear (e.g. --backtracking-args alpha=0.5 t_min=1.0)')
    parser.add_argument('--problem-sizes', nargs='+', type=int, help='sequencia de inteiros expressando os tamanhos de problemas a serem testados (e.g. --problem-sizes 2 5 10 50 100)')
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
    linear_rest = args.linear_rest
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
        linear_rest=linear_rest,
        SEED_NUMPY=seed
    )
    TestingFunctions.run_tests(method, args_dict, problem_sizes = problem_sizes, **kwargs)
