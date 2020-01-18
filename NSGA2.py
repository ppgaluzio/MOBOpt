# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

import array
import random

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

FirstCall = True

def uniform(bounds):
    # N = len(bounds)
    # x = [random.uniform(b[0],b[1]) for b in bounds]
    return [random.uniform(b[0], b[1]) for b in bounds]


def NSGAII(NObj, objective, pbounds, seed=None, NGEN=100, MU=100, CXPB=0.9):
    random.seed(seed)
    # print("IN NSGAII")

    global FirstCall
    if FirstCall:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*NObj)
        creator.create("Individual", array.array, typecode='d',
                       fitness=creator.FitnessMin)
        FirstCall = False
    toolbox = base.Toolbox()

    # Problem definition
    # Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
    # BOUND_LOW, BOUND_UP = 0.0, 1.0

    # Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
    # BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

    # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
    NDIM = len(pbounds)

    toolbox.register("attr_float", uniform, pbounds)

    toolbox.register("individual",
                     tools.initIterate,
                     creator.Individual,
                     toolbox.attr_float)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective)

    toolbox.register("mate",
                     tools.cxSimulatedBinaryBounded,
                     low=pbounds[:,0].tolist(),
                     up=pbounds[:,1].tolist(),
                     eta=20.0)

    toolbox.register("mutate",
                     tools.mutPolynomialBounded,
                     low=pbounds[:,0].tolist(),
                     up=pbounds[:,1].tolist(),
                     eta=20.0,
                     indpb=1.0/NDIM)  # verify //

    toolbox.register("select", tools.selNSGA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    # print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        # print(logbook.stream)

    # print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    front = np.array([ind.fitness.values for ind in pop])

    return pop, logbook, front
