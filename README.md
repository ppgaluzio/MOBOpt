# MOBOpt

Multi-Objective Bayesian Optimization

## Prerequisites

  * Python 3.7
  * numpy 1.16
  * matplotlib 3.0
  * scikit-learn 0.22
  * deap 1.3
  * scipy 1.1

## Instalation

  *  Clone this repo to your local machine using `https://github.com/ppgaluzio/MOBOpt.git`
  *  Add ./MOBOpt to your python path

## Usage

    ``` python
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-

    import numpy as np
    import matplotlib.pyplot as pl
    import MOBOpt.bayes as by
    import deap.benchmarks as db


    def target(x):
        return - np.asarray(db.zdt1(x))  # The method finds the max


    NParam = 2

    PB = np.asarray([[0, 1]]*NParam)

    f1 = np.linspace(0, 1, 1000)
    f2 = (1-np.sqrt(f1))

    Optimize = by.MOBayesianOpt(target=target,
        NObj=2,
        NParam=NParam,
        pbounds=PB,
        Picture=True,
        TPF=np.asarray([f1, f2]).T)

    Optimize.initialize(init_points=5)

    front, pop = Optimize.maximize(n_iter=100,
        prob=0.8, q=0.8,
        ReduceProb=True)

    PF = np.asarray([np.asarray(y) for y in Optimize.y_Pareto])
    PS = np.asarray([np.asarray(x) for x in Optimize.x_Pareto])

    fig, ax = pl.subplots(1, 1)
    ax.plot(f1, f2, '-', label="TPF")
    ax.plot(-PF[:, 0], -PF[:, 1], 'o', label=r"$\chi$")
    ax.scatter(front[:, 0], front[:, 1], label="NSGA")
    ax.grid()
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.legend()
    fig.show()
    ```
