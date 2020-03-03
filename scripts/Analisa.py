#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# See documentation for description of each column

M = pd.read_csv("M.dat",
                sep=' ',
                names=["NDim",
                       "Iter",
                       "N_init",
                       "NPF",
                       "GD",
                       "SS",
                       "HV",
                       "HausDist",
                       "Cover",
                       "GDPS",
                       "SSPS",
                       "HDPS",
                       "Prob",
                       "q",
                       "Front"])

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
