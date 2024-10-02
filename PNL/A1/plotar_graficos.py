from libs import np, sym
import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser
from TestingFunctions import TestingFunctions
#import matplotlib.ticker as ticker
import re

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--func', type=lambda x: getattr(TestingFunctions, x, None))
    parser.add_argument('--size', type=int)
    args = parser.parse_args()
    func = args.func
    func_name = func.__name__
    size = args.size
    func = func(size)
    func = sym.lambdify(list(func.free_symbols), func, 'numpy')

    if func_name == "func_D":
        from matplotlib.colors import LogNorm
        x = np.linspace(-2, 2, 3000)
        y = np.linspace(-1, 3, 3000)
        X, Y = np.meshgrid(x, y)
        Z = func(x0=X,x1=Y)
        plt.xlim(-2, 2)
        plt.ylim(-1, 3)
        a=-4
        b=4
        contour = plt.contourf(X, Y, Z,
            levels=np.logspace(a, b, 250),
            norm=LogNorm(),
            extend="both"
        )

        contour_1 = plt.contour(X, Y, Z, levels=[1e0], colors="white", linewidths=0.5, alpha=0.5)
        contour_0 = plt.contour(X, Y, Z, levels=[1e-2], colors="white", linewidths=0.5, alpha=0.5)
        contour_10 = plt.contour(X, Y, Z, levels=[1e1], colors="white", linewidths=0.5, alpha=0.5)
        contour_2 = plt.contour(X, Y, Z, levels=[1e2], colors="white", linewidths=0.5, alpha=0.5)
        contour_3 = plt.contour(X, Y, Z, levels=[1e3], colors="white", linewidths=0.5, alpha=0.5)
        contour_4 = plt.contour(X, Y, Z, levels=[1e4], colors="white", linewidths=0.5, alpha=0.5)

        levels = np.logspace(a, b, b-a+1)
        plt.scatter(x=np.array([1.0]), y=np.array([1.0]), c="red", s=9.0)
        cbar = plt.colorbar(contour)
        cbar.set_ticks(levels)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.savefig('graph_func_D', dpi=1200, bbox_inches='tight', format='png')
        plt.show()

    elif func_name == "func_E":
        from matplotlib.colors import LogNorm
        x = np.linspace(-4.5, 4.5, 3000)
        y = np.linspace(-4.5, 4.5, 3000)
        X, Y = np.meshgrid(x, y)
        Z = func(x0=X,x1=Y)
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        a=-4
        b=5
        contour = plt.contourf(X, Y, Z,
            levels=np.logspace(a, b, 250),
            norm=LogNorm(),
            extend="both"
        )

        contour_1 = plt.contour(X, Y, Z, levels=[1e0], colors="white", linewidths=0.5, alpha=0.5)
        contour_0 = plt.contour(X, Y, Z, levels=[1e-2], colors="white", linewidths=0.5, alpha=0.5)
        contour_10 = plt.contour(X, Y, Z, levels=[1e1], colors="white", linewidths=0.5, alpha=0.5)
        contour_2 = plt.contour(X, Y, Z, levels=[1e2], colors="white", linewidths=0.5, alpha=0.5)
        contour_3 = plt.contour(X, Y, Z, levels=[1e3], colors="white", linewidths=0.5, alpha=0.5)
        contour_4 = plt.contour(X, Y, Z, levels=[1e4], colors="white", linewidths=0.5, alpha=0.5)

        levels = np.logspace(a, b, b-a+1)
        plt.scatter(x=np.array([3.0]), y=np.array([0.5]), c="red", s=9.0)
        cbar = plt.colorbar(contour)
        cbar.set_ticks(levels)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.savefig('graph_func_E', dpi=1200, bbox_inches='tight', format='png')
        plt.show()

    elif func_name == "func_F":
        from matplotlib.colors import LogNorm
        x = np.linspace(-2, 2, 3000)
        y = np.linspace(-3, 1, 3000)
        X, Y = np.meshgrid(x, y)
        Z = func(x0=X,x1=Y)
        plt.xlim(-2, 2)
        plt.ylim(-3, 1)
        a=0
        b=6
        contour = plt.contourf(X, Y, Z,
            levels=np.logspace(a, b, 250),
            norm=LogNorm(),
            extend="both"
        )

        contour_1 = plt.contour(X, Y, Z, levels=[1e0], colors="white", linewidths=0.5, alpha=0.5)
        contour_10 = plt.contour(X, Y, Z, levels=[1e1], colors="white", linewidths=0.5, alpha=0.5)
        contour_2 = plt.contour(X, Y, Z, levels=[1e2], colors="white", linewidths=0.5, alpha=0.5)
        contour_3 = plt.contour(X, Y, Z, levels=[1e3], colors="white", linewidths=0.5, alpha=0.5)
        contour_4 = plt.contour(X, Y, Z, levels=[1e4], colors="white", linewidths=0.5, alpha=0.5)
        contour_5 = plt.contour(X, Y, Z, levels=[1e5], colors="white", linewidths=0.5, alpha=0.5)
        contour_6 = plt.contour(X, Y, Z, levels=[1e6], colors="white", linewidths=0.5, alpha=0.5)

        levels = np.logspace(a, b, b-a+1)
        plt.scatter(x=np.array([0.0]), y=np.array([-1]), c="red", s=9.0)
        cbar = plt.colorbar(contour)
        cbar.set_ticks(levels)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.savefig('graph_func_F', dpi=1200, bbox_inches='tight', format='png')
        plt.show()


    elif func_name == "func_G":
        from matplotlib.colors import LogNorm
        x = np.linspace(-4.5, 4.5, 3000)
        y = np.linspace(-4.5, 4.5, 3000)
        X, Y = np.meshgrid(x, y)
        Z = func(x0=X,x1=Y)
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        a=-3
        b=3
        contour = plt.contourf(X, Y, Z,
            levels=np.logspace(a, b, 250),
            norm=LogNorm(),
            extend="both"
        )

        contour_1 = plt.contour(X, Y, Z, levels=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], colors="white", linewidths=0.5, alpha=0.5)

        levels = np.logspace(a, b, b-a+1)
        plt.scatter(x=np.array([3.0]), y=np.array([2.0]), c="red", s=9.0)
        plt.scatter(x=np.array([-2.805118]), y=np.array([3.131312]), c="red", s=9.0)
        plt.scatter(x=np.array([-3.779310]), y=np.array([-3.283186]), c="red", s=9.0)
        plt.scatter(x=np.array([3.584428]), y=np.array([-1.848126]), c="red", s=9.0)
        cbar = plt.colorbar(contour)
        cbar.set_ticks(levels)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.savefig('graph_func_G', dpi=1200, bbox_inches='tight', format='png')
        plt.show()