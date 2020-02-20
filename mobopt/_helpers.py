# -*- coding: utf-8 -*-

import matplotlib.pyplot as pl


# % Clip x between xmin and xmax
def clip(x, xmin, xmax):
    for i, xx in enumerate(x):
        if xmin[i] is not None:
            if xx < xmin[i]:
                x[i] = xmin[i]
        if xmax[i] is not None:
            if xx > xmax[i]:
                x[i] = xmax[i]
    return


# % Visualiza
def plot_1dgp(fig, ax, space, iterations, Front, last):

    ax.clear()

    PF = space.f

    lineObs, = ax.plot(-PF[:, 0], -PF[:, 1], 'o', label=f"N = {iterations}",
                       alpha=0.2)
    lineNsga, = ax.plot(Front[:, 0], Front[:, 1], 'o', label="NSGA", alpha=0.5)
    Last, =  ax.plot(Front[last, 0], Front[last, 1], '*')
    lastObs, = ax.plot(-PF[-1, 0], -PF[-1, 1], '*')

    ax.grid()
    pl.xlabel('$f_1$')
    pl.ylabel('$f_2$')
    pl.legend(loc='upper right')

    fig.canvas.draw()
    fig.canvas.flush_events()
