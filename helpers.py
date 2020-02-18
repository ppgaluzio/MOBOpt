# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

import contextlib
import io
import sys

#%% supress output

@contextlib.contextmanager
def no_out():
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.BytesIO()
    sys.stderr = io.BytesIO()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr

# temporary definition
# @contextlib.contextmanager
# def no_out():
#     yield

#%% Clip x between xmin and xmax

def clip(x,xmin,xmax) :
    for i, xx in enumerate(x):
        if xmin[i] != None :
            if xx < xmin[i] :
                x[i] = xmin[i]
        if xmax[i] != None :
            if xx > xmax[i] :
                x[i] = xmax[i]
    return

#%% Visualiza

def plot_1dgp(fig,ax,space,iterations,Front,last) :

    ax.clear()

    # y_Pareto, x_Pareto = space.ParetoSet()
    # PF = np.asarray([np.asarray(y) for y in y_Pareto])
    PF = space.f

    # f1 = np.linspace(0,1,1000)
    # f2 = (1-np.sqrt(f1))
    # TPF, = ax.plot(f1,f2,'-',label="TPF")

    # newObs,  = ax.plot(-space._F[space.length-1][0],
    #                    -space._F[space.length-1][1],'o')
    lineObs, = ax.plot(-PF[:,0],-PF[:,1],'o',label="N = %d" % iterations,alpha=0.2)
    lineNsga, = ax.plot(Front[:,0], Front[:,1], 'o',label="NSGA",alpha=0.5)
    Last, =  ax.plot(Front[last,0], Front[last,1], '*')
    lastObs, = ax.plot(-PF[-1,0],-PF[-1,1],'*')


    # J = np.empty(PF.shape[0])
    # for i in range(PF.shape[0]) :
    #     J[i] = np.absolute(np.asarray(PF[i,:]))[0]

    # fator = [1,1,1,1,-1,-1]

    # i = 0
    # for index in np.ndindex(ax.shape) :
    #     lineObs, = ax[index].plot(fator[i] * PF[:,i],J,'o')
    #     newObs,  = ax[index].plot(fator[i] * space._F[space.length-1][i],J[-1],'o')
    #     ax[index].grid()
    #     i += 1

    ax.grid()
    # ax.tick_params(labelbottom=False,labelleft=False)
    # pl.xlabel('$f_1$')
    # pl.ylabel('$f_2$')
    # pl.ylim(MIN-epss, MAX+epss)
    pl.legend(loc='upper right')

    fig.canvas.draw()
    fig.canvas.flush_events()
