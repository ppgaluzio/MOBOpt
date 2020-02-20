# -*- coding: utf-8 -*-

import numpy as np


def GD(S, P):
    """
    S -> Solution set by the algorithm
    P -> Pareto front (reference set)
    """
    NS = len(S)
    d = np.empty(NS)

    for i in range(NS):
            d[i] = np.asarray(np.sqrt((S[i, :]-P)**2)).min()

    gd = np.sqrt(np.sum(d**2)) / NS
    return gd


def __DistF(p, fs, P):
    d2 = np.sum((fs-P(p))**2)
    return np.sqrt(d2)


def Spread2D(S, P):
    """
    S -> Solution set by the algorithm
    P -> Pareto front (reference set)
    """
    NS = len(S)
    S = np.sort(S, axis=0)
    P = np.sort(P, axis=0)

    df = np.sqrt((S[:, 0]-P[0, 0])**2+(S[:, 1]-P[0, 1])**2).min()
    dl = np.sqrt((S[:, 0]-P[-1, 0])**2+(S[:, 1]-P[-1, 1])**2).min()

    d = np.sqrt((S[:-1, 0]-S[1:, 0])**2+(S[:-1, 1]-S[1:, 1])**2)
    dm = np.average(d)

    Delta = (df + dl + np.sum(np.abs(d-dm))) / (df + dl + (NS-1)*dm)

    return Delta


def Coverage(S, NDiv=100):
    NObj = S.shape[1]
    Cover = np.zeros(NObj)
    for iObj in range(NObj):
        F = np.sort(S[:, iObj])
        Counter = np.zeros(NDiv, dtype=bool)
        Delta = (F[-1]-F[0])/NDiv
        # print("DELTA = ", Delta)
        iDiv = 0
        for ff in F:
            # print("FF = ", ff, F[0], F[-1])
            while (ff > (F[0] + (iDiv + 1) * Delta)):
                iDiv += 1
            try:
                Counter[iDiv] = True
            except IndexError:
                Counter[-1] = True

        Cover[iObj] = Counter.sum()/NDiv
    return Cover.mean()
