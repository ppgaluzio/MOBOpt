# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as pl

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from .target_space import TargetSpace
from .helpers import no_out, plot_1dgp
# import types

from .NSGA2 import NSGAII
# import deap.benchmarks as db
from deap.benchmarks.tools import hypervolume
from .metrics import GD, Spread2D, Coverage
from scipy.spatial.distance import directed_hausdorff as HD


# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Class Bayesians Optimization


class MOBayesianOpt(object):

    def __init__(self, target, NObj, NParam, pbounds, constraints=[],
                 verbose=False, Picture=True, TPF=None,
                 n_restarts_optimizer=1000, Filename="M", MetricsPS=True,
                 FrontSampling=[10, 25, 50, 100]):
        """
        Bayesian optimization object

        Keyword Arguments:
        target  -- list of functions to be optimized
        NObj    -- Number of objective functions
        NParam  -- Number of parameters for the objective function
        pbounds -- numpy array with bounds for each parameter
                   pbounds.shape == (NParam,2)
        verbose -- Whether or not to print progress (default False)

        Based heavily on github.com/fmfn/BayesianOptimization
        """

        super(MOBayesianOpt, self).__init__()

        self.verbose = verbose
        self.vprint = print if verbose else lambda *a, **k: None
        self.FrontSampling = FrontSampling
        # set counter
        self.counter = 0
        self.constraints = constraints
        self.Picture = Picture
        self.n_rest_opt = n_restarts_optimizer
        self.Filename = Filename
        self.MetricsPS = MetricsPS

        # reset calling variables
        self.__reset__()

        # number of objective functions
        if isinstance(NObj, int):
            self.NObj = NObj
        else:
            raise TypeError("NObj should be int")

        # objective function returns lists w/ the multiple target functions
        if callable(target):
            self.target = target
        else:
            raise TypeError("target should be callable")

        # number of parameters
        if isinstance(NParam, int):
            self.NParam = NParam
        else:
            raise TypeError("NParam should be int")

        # print("NParam = ",self.NParam)

        if TPF is None:
            self.vprint("no metrics are going to be saved")
            self.Metrics = False
        else:
            self.vprint("metrics are going to be saved")
            self.Metrics = True
            self.TPF = TPF

        self.vprint("Filename = "+self.Filename+".dat")
        self.FF = open(Filename+".dat", "a", 1)
        self.vprint("Saving:")
        self.vprint("NParam, iter, N init, NFront,"
                    "GenDist, SS, HV, HausDist, Cover, GDPS, SSPS,"
                    "HDPS, NewProb, q, FrontFilename")

        # pbounds must hold the bounds for each parameter
        try:
            if len(pbounds) == self.NParam:
                self.pbounds = pbounds
            else:
                raise IndexError("pbounds must have dimension equal to NParam")
        except:
            raise TypeError("pbounds is neither a np.array nor a list")
        if self.pbounds.shape != (NParam,2) :
            raise IndexError("pbounds must have 2nd dimension equal to 2")

        with no_out():
            self.GP = [None] * self.NObj
            for i in range(self.NObj) :
                self.GP[i] = GPR(kernel=Matern(nu=1.5),
                                 n_restarts_optimizer = self.n_rest_opt)

        # store starting points
        self.init_points = []

        self.space = TargetSpace(self.target,self.NObj,self.pbounds,
                                 self.constraints,verbose=self.verbose)

        if self.Picture: #and self.NObj == 2:
            self.fig,self.ax = pl.subplots(1,1,figsize=(5,4))
            self.fig.show()

        return


    #%% RESET

    def __reset__(self):
        """
        RESET all function initialization variables
        """
        self.__CalledInit = False

        return

    #%% INIT

    def initialize(self,init_points = None,Points = []):
        """
        Initialization of the method

        Keyword Arguments:
        init_points -- Number of random points to probe

        At first, no points provided by the user are gonna be used by the algorithm
        Only points calculated randomly, respecting the bounds provided
        """

        self.N_init_points = 0
        if init_points != None :
            self.N_init_points += init_points

            # initialize first points for the gp fit,
            # random points respecting the bounds of the variables.
            rand_points = self.space.random_points(init_points)
            self.init_points.extend(rand_points)
            self.init_points = np.asarray(self.init_points)

            # evaluate target function at all intialization points
            for x in self.init_points :
                y = self.space.observe_point(x)

        if len(Points) > 0 :
            for ii in range(len(Points)) :
                xx = Points[ii]
                y = self.space.observe_point(Points[ii])
                self.N_init_points += 1

        self.vprint("Added points in init")
        self.vprint(self.space.x)

        self.__CalledInit = True

        return

    #%% maximize

    def maximize(self,
                 init_points=5,
                 n_iter=100,
                 prob=0.1,
                 ReduceProb = False,
                 q = 0.5,
                 **gp_params):

        # allocate necessary memory

        # If initialize was not called, call it and allocate necessary space
        if not self.__CalledInit:
            self.initialize(init_points)
            self.space._allocate(init_points+n_iter)

        # Else, just allocate the necessary space
        else:
            if self.N_init_points+n_iter > self.space._n_alloc_rows :
                self.space._allocate(self.N_init_points+n_iter)

        self.q = q
        self.NewProb = prob

        self.vprint("Start optimization loop")

        for i in range(n_iter) :

            self.vprint(i," of ",n_iter)
            if ReduceProb:
                self.NewProb = prob * (1.0 - self.counter/n_iter)

            with no_out():
                for i in range(self.NObj):
                    yy = self.space.f[:, i]
                    self.GP[i].fit(self.space.x, yy)
            pop, logbook, front = NSGAII(self.NObj,
                                         self.ObjectiveGP,
                                         self.pbounds)

            Population = np.asarray(pop)
            IndexF, FatorF = self.LargestOfLeast(front, self.space.f)
            IndexPop, FatorPop = self.LargestOfLeast(Population, self.space.x)

            Fator = self.q * FatorF + (1-self.q) * FatorPop
            Index_try = np.argmax(Fator)

            self.vprint("IF = ", IndexF,
                        " IP = ", IndexPop,
                        " Try = ", Index_try)

            self.vprint("Front at = ", -front[Index_try])

            self.x_try = Population[Index_try]

            if self.Picture:
                plot_1dgp(fig=self.fig, ax=self.ax, space=self.space,
                          iterations=self.counter+len(self.init_points),
                          Front=front, last=Index_try)

            if np.random.uniform() < self.NewProb:

                if self.NParam > 1:
                    ii = np.random.randint(low=0,high=self.NParam - 1)
                else:
                    ii = 0

                self.x_try[ii] = np.random.uniform(low = self.pbounds[ii][0],
                                                   high = self.pbounds[ii][1])

                self.vprint("Random Point at ",ii," coordinate")

            y = self.space.observe_point(self.x_try)

            self.y_Pareto, self.x_Pareto = self.space.ParetoSet()
            self.counter += 1

            self.vprint ("|PF| = {:4d} at {:4d} of {:4d}, w/ k = {:4.2f}"\
                         .format(self.space.ParetoSize,
                                 self.counter,
                                 n_iter,
                                 self.NewProb))

            for NFront in self.FrontSampling:
                if (self.counter % 10 == 0) and (NFront == self.FrontSampling[-1]):
                    SaveFile = True
                else:
                    SaveFile = False
                Ind = np.random.choice(front.shape[0], NFront, replace=False)
                PopInd = [pop[i] for i in Ind]
                self.PrintOutput(front[Ind, :], PopInd,
                                 SaveFile)


        return front, np.asarray(pop)

    def LargestOfLeast(self,front,F) :
        NF = len(front)
        MinDist = np.empty(NF)
        for i in range(NF) :
            MinDist[i] =  self.MinimalDistance(-front[i],F)
        # print ("MIN DIST = ",MinDist)
        ArgMax = np.argmax(MinDist)
        # print ("ARGMAX = ",ArgMax," w/ ",MinDist[ArgMax])
        Mean = MinDist.mean()
        Std = np.std(MinDist)
        return ArgMax, (MinDist-Mean)/(Std)

    def PrintOutput(self, front, pop, SaveFile=False):

        NFront = front.shape[0]

        if self.Metrics:
            GenDist = GD(front, self.TPF)
            SS = Spread2D(front, self.TPF)
            HausDist = HD(front, self.TPF)[0]
        else:
            GenDist = np.nan
            SS = np.nan
            HausDist = np.nan

        Cover = Coverage(front)
        HV = hypervolume(pop, [11.0]*self.NObj)

        if self.MetricsPS and self.Metrics:
            FPS = []
            for x in pop:
                FF = - self.target(x)
                FPS += [[FF[i] for i in range(self.NObj)]]
            FPS = np.array(FPS)

            GDPS = GD(FPS, self.TPF)
            SSPS = Spread2D(FPS, self.TPF)
            HDPS = HD(FPS, self.TPF)[0]
        else:
            GDPS = np.nan
            SSPS = np.nan
            HDPS = np.nan


        self.vprint("NFront = {}, GD = {:7.3e} | SS = {:7.3e} | HV = {:7.3e} "\
                    .format(NFront,GenDist, SS, HV))

        if SaveFile:
            FrontFilename = "FF_D{:02d}_I{:04d}_NI{:02d}_P{:4.2f}_Q{:4.2f}".format(self.NParam,self.counter,self.N_init_points,self.NewProb,self.q)+self.Filename

            # "Front{:03d}_".format(self.counter)+self.Filename
            # np.savetxt(FrontFilename,front)


            PF = np.asarray([np.asarray(y) for y in self.y_Pareto])
            PS = np.asarray([np.asarray(x) for x in self.x_Pareto])

            Population = np.asarray(pop)
            np.savez(FrontFilename,
                     Front = front,
                     Pop = Population,
                     PF = PF,
                     PS = PS)

            FrontFilename += ".npz"
        else :
            FrontFilename = np.nan

        # try:
        self.FF.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n"\
                      .format(self.NParam,
                              self.counter+len(self.init_points),
                              self.N_init_points,
                              NFront,
                              GenDist,
                              SS,
                              HV,
                              HausDist,
                              Cover,
                              GDPS,
                              SSPS,
                              HDPS,
                              self.NewProb,
                              self.q,
                              FrontFilename))
        # except:
        #     print("DID NOT PRINT")
        #     pass

        return


    @staticmethod
    def MinimalDistance(X,Y) :
        N = len(X)
        Npts = len(Y)
        DistMin = float('inf')
        for i in range(Npts) :
            Dist = 0.
            for j in range(N) :
                Dist += (X[j]-Y[i,j])**2
            Dist = np.sqrt(Dist)
            # Dist = np.absolute(X-Y[i,:]).max()
            if Dist < DistMin :
                DistMin = Dist
        return DistMin

    def MaxDist(self,front,yPareto):
        NF = len(front)
        IndexMax = 0
        DistMax = self.DistTotal(-front[0],yPareto)
        for i in range(1,NF) :
            Dist = self.DistTotal(-front[i],yPareto)
            if Dist > DistMax :
                DistMax = Dist
                IndexMax = i
        return IndexMax

    @staticmethod
    def DistTotal(X,Y):
        Soma  = 0.0
        for i in range(len(Y)):
            Dist = 0.0
            for j in range(len(X)) :
                Dist += (X[j]-Y[i,j])**2
            Dist = np.sqrt(Dist)
            Soma += Dist
        return Soma / len(Y)




    #%% Define the function to be optimized by nsga2


    def ObjectiveGP(self,x):

        Fator = 1.0e10
        F = [None] * self.NObj
        xx = np.asarray(x).reshape(1,-1)

        Constraints = 0.0
        for cons in self.constraints :
            y = cons['fun'](x)
            if cons['type'] == 'eq' :
                Constraints += np.abs(y)
            elif cons['type'] == 'ineq' :
                if y < 0 :
                    Constraints -= y

        for i in range(self.NObj) :
            # Fator = 10.0 * np.max(self.space.f[:,i])
            F[i] = -self.GP[i].predict(xx)[0] + Fator * Constraints

        return F


    #%% Sigmoid
    @staticmethod
    def Sigmoid(x,k=10.) :
        return 1./(1.+np.exp(k*(x-0.5)))


    #%% write relevant information to file

    def WriteSpace(self,filename="space"):

        Info = [self.space.NObj,
                self.space.NParam,
                self.space._NObs,
                self.space.length]

        np.savez(filename,
                 X = self.space._X,
                 Y = self.space._Y,
                 F = self.space._F,
                 I = Info)

        return

    #%% read relevant information from file

    def ReadSpace(self, filename="space.npz"):

        Data = np.load(filename)

        self.space.NObj   = Data["I"][0]
        self.space.NParam = Data["I"][1]
        self.space._NObs  = Data["I"][2]
        self.space.length = Data["I"][3]

        self.space._allocate((self.space.length + 1) * 2)

        self.space._X = Data["X"]
        self.space._Y = Data["Y"]
        self.space._F = Data["F"]

        # Redefine GP
        with no_out():
        # internal GP regressor
            self.GP = [None] * self.NObj
            for i in range(self.NObj) :
                self.GP[i] = GPR(kernel=Matern(nu=0.5),
                                 n_restarts_optimizer = self.n_rest_opt)

        with no_out():
            for i in range(self.NObj) :
                yy = self.space.f[:,i]
                self.GP[i].fit(self.space.x,yy)


        return
