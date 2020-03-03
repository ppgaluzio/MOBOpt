#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

M = pd.read_csv("M.dat",
                sep=' ',
                names=["NDim",  # Dimensionalidade do espaço de busca
                       "Iter",  # Número de avaliações da função objetivo
                       "N_init",  # Pontos iniciais
                       "NPF",     # N. de pts na PF
                       "GD",      # Generational distance
                       "SS",      # Spread
                       "HV",      # Hyervolume
                       "HausDist",  # Haussdorf Distance
                       "Cover",     # Coverage
                       "GDPS",      # GD measured by PS
                       "SSPS",      # Spread measured by PS
                       "HDPS",  # Haussdorf distance measured by PS
                       "Prob",  # Probability
                       "q",     # Q
                       "Front"])  # Front Filename

ZG = {}
Keys = []

Metric = "GD"

fig = pl.figure(Metric)

ZG = M.\
    groupby("NPF").get_group(100).\
    groupby("Iter").agg(np.mean).reset_index()

pl.plot(ZG["Iter"], ZG[Metric], 'o')
print(ZG[Metric])

pl.legend()
pl.grid()
pl.xscale("log")
pl.yscale("log")
fig.show()
