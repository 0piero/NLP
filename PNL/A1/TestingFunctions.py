from .libs import sym, np, time, pickle
LOW_RANDOM_POINT=-10
HIGH_RANDOM_POINT=10


class TestingFunctions(object):
    def func_A(n):
        x = sym.symbols(f'x:{n}')
        y = 0
        for i in range(1, n+1):
            y += 1.01**i * x[i-1]**2
        return y

    def func_B(n):
        x = sym.symbols(f'x:{n}')
        y = 0
        for i in range(1, n+1):
            y += 1.1**i * x[i-1]**2
        return y

    def func_C(n):
        x = sym.symbols(f'x:{n}')
        y = 0
        for i in range(1, int(n/2) + 1):
            y += 1.01**i * x[i-1]**2

        for i in range(int(n/2) + 1, n+1):
            y += (20.0 + 1.01**i) * x[i-1]**2
        return y

    def func_D(n):
        x = sym.symbols(f'x:{n}')
        y = 0
        for i in range(1, n):
            y += 100*(x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2

        return y

    def func_E(*args):
        x = sym.symbols(f'x:{2}')
        y = (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

        return y

    def func_F(*args):
        x = sym.symbols(f'x:{2}')
        y = (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) * (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
        return y

    def func_G(*args):
        x = sym.symbols(f'x:{2}')
        y = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
        return y

    def run_tests(method: "IterativeGradient", method_args, problem_sizes, **kwargs):

        for size in problem_sizes:
            kwargs.update({"size": size})
            m = method(**kwargs)
            args = method_args.copy()
            args["f"] = method_args["f"](size)
            args["x0"] = method_args["x0"](low=LOW_RANDOM_POINT, high=HIGH_RANDOM_POINT, size=(size,))
            if args["linear_rest"]:
                args["x0"] = args["x0"] / args["x0"].sum()
            SEED_NUMPY = args["SEED_NUMPY"]
            del args["SEED_NUMPY"]
            if size<=2:
                args["record_iterands"] = True
            start_time = time.process_time()
            m.iterative_gradient(**args)
            end_time = time.process_time()
            execution_time = end_time - start_time
            scores = m.get_experiment_scores()
            file_name = '[{}]_[{}_size_{}]_[{}]_[{}]'.format(
                method.__name__,
                method_args["f"].__name__,
                size, args["newton_args"],
                args["backtracking_args"]
            )
            with open(file_name + "_seed{}".format(SEED_NUMPY), "w") as file:

                file.write(f"Execution time for {method.__name__}: {execution_time} s\n")
                file.write("max_iter: {}\n".format(args["max_iter"]))
                file.write("newton_args: {}\n".format(args["newton_args"]))
                file.write("backtracking_args: {}\n".format(args["backtracking_args"]))
                file.write("convergence_accepted: {}\n".format(scores["convergence_accepted"]))
                file.write("total_iterations: {}\n".format(scores["total_iterations"]))

            with open(f'{file_name}_experiment_scores_seed{SEED_NUMPY}.pickle', 'wb') as f:
                pickle.dump(scores, f)

