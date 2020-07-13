# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

# from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from scipy.spatial.distance import directed_hausdorff as HD
from deap.benchmarks.tools import hypervolume
from warnings import warn

from ._wrapper import GaussianProcessWrapper as GPR
from ._NSGA2 import NSGAII
from .metrics import GD, Spread2D, Coverage
from ._target_space import TargetSpace
from ._helpers import plot_1dgp


class ConstraintError(Exception):
    pass


def max_or_min_wrapper(function, max_or_min):
    if max_or_min == 'max':
        def fun_wrapper(*wrapper_args):
            return function(*(wrapper_args))
    elif max_or_min == 'min':
        def fun_wrapper(*wrapper_args):
            return - function(*(wrapper_args))
    else:
        raise ValueError("max_or_min should be either 'max' or 'min'")
    return fun_wrapper


# Class Bayesians Optimization
class MOBayesianOpt(object):

    def __init__(self, target, NObj, pbounds, constraints=[],
                 verbose=False, Picture=False, TPF=None,
                 n_restarts_optimizer=10, Filename=None,
                 MetricsPS=True, max_or_min='max', RandomSeed=None,
                 kernel=None):
        """Bayesian optimization object

        Keyword Arguments:
        target  -- functions to be optimized
                   def target(x): x is a np.array
                       return [f_1, f_2, ..., f_NObj]

        NObj    -- int, Number of objective functions

        pbounds -- numpy array with bounds for each parameter
                   pbounds.shape == (NParam,2)

        constraints -- list of dictionary with constraints
                   [{'type': 'ineq', 'fun': constr_fun}, ...]

                   def constr_fun(x):
                       return g(x) # >= 0

        verbose -- Whether or not to print progress (default False)

        Picture -- bool (default True)
                   whether or not to plot PF convergence, for NObj = 2 only

        TPF -- np.ndarray (default None)
               Array with the True Pareto Front for calculation of
               convergence metrics

        n_restarts_optimizer -- int (default 100)
             GP parameter, the number of restarts of the optimizer for
             finding the kernel’s parameters which maximize the log-marginal
             likelihood.

        Filename -- string (default None)
             Partial metrics will be
             saved at filename, if None nothing is saved

        MetricsPS -- bool (default True)
             whether os not to calculate metrics with the Pareto Set points

        max_or_min -- str (default 'max')
             whether the optimization problem is a maximization
             problem ('max'), or a minimization one ('min')

        RandomSeed -- {None, int, array_like}, optional
            Random seed used to initialize the pseudo-random number
            generator. Can be any integer between 0 and 2**32 - 1
            inclusive, an array (or other sequence) of such integers,
            or None (the default). If seed is None, then RandomState
            will try to read data from /dev/urandom (or the Windows
            analogue) if available or seed from the clock otherwise.

        kernel -- kernel object
            kernel object to be passed to the gausian process
            regressor, if None, the default `Matern(nu=1.5)` is used

            For valid kernel objects, visit:
            https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes)

        Based heavily on github.com/fmfn/BayesianOptimization

        """

        super(MOBayesianOpt, self).__init__()

        self.verbose = verbose
        self.vprint = print if verbose else lambda *a, **k: None

        self.counter = 0
        self.constraints = constraints
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

        if Picture and self.NObj == 2:
            self.Picture = Picture
        else:
            if Picture:
                warn("NObj must be 2 to plot PF convergence")
            self.Picture = False

        # objective function returns lists w/ the multiple target functions
        if callable(target):
            self.target = max_or_min_wrapper(target, max_or_min)
        else:
            raise TypeError("target should be callable")

        self.pbounds = pbounds
        # pbounds must hold the bounds for each parameter
        try:
            self.NParam = len(pbounds)
        except TypeError:
            raise TypeError("pbounds is neither a np.array nor a list")
        if self.pbounds.shape != (self.NParam, 2):
            raise IndexError("pbounds must have 2nd dimension equal to 2")

        self.vprint(f"Dim. of Search Space = {self.NParam}")

        if TPF is None:
            self.vprint("no metrics are going to be saved")
            self.Metrics = False
        else:
            self.vprint("metrics are going to be saved")
            self.Metrics = True
            self.TPF = TPF

        if self.Filename is not None:
            self.__save_partial = True
            self.vprint("Filename = "+self.Filename)
            self.FF = open(Filename, "a", 1)
            self.vprint("Saving:")
            self.vprint("NParam, iter, N init, NFront,"
                        "GenDist, SS, HV, HausDist, Cover, GDPS, SSPS,"
                        "HDPS, NewProb, q, FrontFilename")
        else:
            self.__save_partial = False

        if kernel is None:
            kernel = Matern(nu=1.5)

        self.GP = [None] * self.NObj
        for i in range(self.NObj):
            self.GP[i] = GPR(kernel=kernel,
                             n_restarts_optimizer=self.n_rest_opt)

        # store starting points
        self.init_points = []

        # test for constraint types
        for cc in self.constraints:
            if cc['type'] == 'eq':
                raise ConstraintError(
                    "Equality constraints are not implemented")

        self.space = TargetSpace(self.target, self.NObj, self.pbounds,
                                 self.constraints,
                                 RandomSeed=RandomSeed,
                                 verbose=self.verbose)

        if self.Picture and self.NObj == 2:
            self.fig, self.ax = pl.subplots(1, 1, figsize=(5, 4))
            self.fig.show()

        return

    # % RESET
    def __reset__(self):
        """
        RESET all function initialization variables
        """
        self.__CalledInit = False

        return

    # % INIT
    def initialize(self, init_points=None, Points=None, Y=None):
        """Initialization of the method

        Keyword Arguments:
        init_points -- Number of random points to probe
        points -- (optional) list of points in which to sample the method
        Y -- (optional) list values of the objective function at the
            points in `Points`. If not provided the method evaluates
            the objective at the points in `Points`

        At first, no points provided by the user are gonna be used by the
        algorithm, Only points calculated randomly, respecting the bounds
        provided

        """

        self.N_init_points = 0
        if init_points is not None:
            self.N_init_points += init_points

            # initialize first points for the gp fit,
            # random points respecting the bounds of the variables.
            rand_points = self.space.random_points(init_points)
            self.init_points.extend(rand_points)
            self.init_points = np.asarray(self.init_points)

            # evaluate target function at all intialization points
            for x in self.init_points:
                self.space.observe_point(x)

        if Points is not None:
            if Y is None:
                for x in Points:
                    self.space.observe_point(np.array(x))
                    self.N_init_points += 1
            else:
                for x, y in zip(Points, Y):
                    self.space.add_observation(np.array(x), np.array(y))
                    self.N_init_points += 1

        if self.N_init_points == 0:
            raise RuntimeError(
                "A non-zero number of initialization points is required")

        self.vprint("Added points in init")
        self.vprint(self.space.x)

        self.__CalledInit = True

        return

    # % maximize
    def maximize(self,
                 n_iter=100,
                 prob=0.1,
                 ReduceProb=False,
                 q=0.5,
                 n_pts=100,
                 SaveInterval=10,
                 FrontSampling=[10, 25, 50, 100]):
        """
        maximize

        input
        =====

        n_iter -- int (default 100)
            number of iterations of the method

        prob -- float ( 0 < prob < 1, default 0.1
            probability of chosing next point randomly

        ReduceProb -- bool (default False)
            if True prob is reduced to zero along the iterations of the method

        q -- float ( 0 < q < 1.0, default 0.5 )
            weight between Search space and objective space when selecting next
            iteration point
            q = 1 : objective space only
            q = 0 : search space only

        n_pts -- int
            effective size of the pareto front
            (len(front = n_pts))

        SaveInterval -- int
            at every SaveInterval save a npz file with the full pareto front at
            that iteration

        FrontSampling -- list of ints
             Number of points to sample the pareto front for metrics

        return front, pop
        =================

        front -- Pareto front of the method as found by the nsga2 at the
                 last iteration of the method
        pop -- population of points in search space as found by the nsga2 at
               the last iteration of the method

        Outputs
        =======

        self.y_Pareto :: list of non-dominated points in objective space
        self.x_Pareto :: list of non-dominated points in search space
        """
        # If initialize was not called, call it and allocate necessary space
        if not self.__CalledInit:
            raise RuntimeError("Initialize was not called, "
                               "call it before calling maximize")

        if not isinstance(n_iter, int):
            raise TypeError(f"n_iter should be int, {type(n_iter)} instead")

        if not isinstance(n_pts, int):
            raise TypeError(f"n_pts should be int, "
                            f"{type(n_pts)} instead")

        if not isinstance(SaveInterval, int):
            raise TypeError(f"SaveInterval should be int, "
                            f"{type(SaveInterval)} instead")

        if isinstance(FrontSampling, list):
            if not all([isinstance(n, int) for n in FrontSampling]):
                raise TypeError(f"FrontSampling should be list of int")
        else:
            raise TypeError(f"FrontSampling should be a list")

        if not isinstance(prob, (int, float)):
            raise TypeError(f"prob should be float, "
                            f"{type(prob)} instead")

        if not isinstance(q, (int, float)):
            raise TypeError(f"q should be float, "
                            f"{type(q)} instead")

        if not isinstance(ReduceProb, bool):
            raise TypeError(f"ReduceProb should be bool, "
                            f"{type(ReduceProb)} instead")

        # Allocate necessary space
        if self.N_init_points+n_iter > self.space._n_alloc_rows:
            self.space._allocate(self.N_init_points+n_iter)

        self.q = q
        self.NewProb = prob

        self.vprint("Start optimization loop")

        for i in range(n_iter):

            self.vprint(i, " of ", n_iter)
            if ReduceProb:
                self.NewProb = prob * (1.0 - self.counter/n_iter)

            for i in range(self.NObj):
                yy = self.space.f[:, i]
                self.GP[i].fit(self.space.x, yy)

            pop, logbook, front = NSGAII(self.NObj,
                                         self.__ObjectiveGP,
                                         self.pbounds,
                                         MU=n_pts)

            Population = np.asarray(pop)
            IndexF, FatorF = self.__LargestOfLeast(front, self.space.f)
            IndexPop, FatorPop = self.__LargestOfLeast(Population,
                                                       self.space.x)

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

            if self.space.RS.uniform() < self.NewProb:

                if self.NParam > 1:
                    ii = self.space.RS.randint(low=0, high=self.NParam - 1)
                else:
                    ii = 0

                self.x_try[ii] = self.space.RS.uniform(
                    low=self.pbounds[ii][0],
                    high=self.pbounds[ii][1])

                self.vprint("Random Point at ", ii, " coordinate")

            dummy = self.space.observe_point(self.x_try)  # noqa

            self.y_Pareto, self.x_Pareto = self.space.ParetoSet()
            self.counter += 1

            self.vprint(f"|PF| = {self.space.ParetoSize:4d} at"
                        f" {self.counter:4d}"
                        f" of {n_iter:4d}, w/ r = {self.NewProb:4.2f}")

            if self.__save_partial:
                for NFront in FrontSampling:
                    if (self.counter % SaveInterval == 0) and \
                       (NFront == FrontSampling[-1]):
                        SaveFile = True
                    else:
                        SaveFile = False
                    Ind = self.space.RS.choice(front.shape[0], NFront,
                                               replace=False)
                    PopInd = [pop[i] for i in Ind]
                    self.__PrintOutput(front[Ind, :], PopInd,
                                       SaveFile)

        return front, np.asarray(pop)

    def __LargestOfLeast(self, front, F):
        NF = len(front)
        MinDist = np.empty(NF)
        for i in range(NF):
            MinDist[i] = self.__MinimalDistance(-front[i], F)

        ArgMax = np.argmax(MinDist)

        Mean = MinDist.mean()
        Std = np.std(MinDist)
        return ArgMax, (MinDist-Mean)/(Std)

    def __PrintOutput(self, front, pop, SaveFile=False):

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

        self.vprint(f"NFront = {NFront}, GD = {GenDist:7.3e} |"
                    f" SS = {SS:7.3e} | HV = {HV:7.3e} ")

        if SaveFile:
            FrontFilename = f"FF_D{self.NParam:02d}_I{self.counter:04d}_" + \
                f"NI{self.N_init_points:02d}_P{self.NewProb:4.2f}_" + \
                f"Q{self.q:4.2f}" + \
                self.Filename

            PF = np.asarray([np.asarray(y) for y in self.y_Pareto])
            PS = np.asarray([np.asarray(x) for x in self.x_Pareto])

            Population = np.asarray(pop)
            np.savez(FrontFilename,
                     Front=front,
                     Pop=Population,
                     PF=PF,
                     PS=PS)

            FrontFilename += ".npz"
        else:
            FrontFilename = np.nan

        self.FF.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n"
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

        return

    @staticmethod
    def __MinimalDistance(X, Y):
        N = len(X)
        Npts = len(Y)
        DistMin = float('inf')
        for i in range(Npts):
            Dist = 0.
            for j in range(N):
                Dist += (X[j]-Y[i, j])**2
            Dist = np.sqrt(Dist)
            if Dist < DistMin:
                DistMin = Dist
        return DistMin

    def __MaxDist(self, front, yPareto):
        NF = len(front)
        IndexMax = 0
        DistMax = self.__DistTotal(-front[0], yPareto)
        for i in range(1, NF):
            Dist = self.__DistTotal(-front[i], yPareto)
            if Dist > DistMax:
                DistMax = Dist
                IndexMax = i
        return IndexMax

    @staticmethod
    def __DistTotal(X, Y):
        Soma = 0.0
        for i in range(len(Y)):
            Dist = 0.0
            for j in range(len(X)):
                Dist += (X[j]-Y[i, j])**2
            Dist = np.sqrt(Dist)
            Soma += Dist
        return Soma / len(Y)

    # % Define the function to be optimized by nsga2

    def __ObjectiveGP(self, x):

        Fator = 1.0e10
        F = [None] * self.NObj
        xx = np.asarray(x).reshape(1, -1)

        Constraints = 0.0
        for cons in self.constraints:
            y = cons['fun'](x)
            if cons['type'] == 'eq':
                Constraints += np.abs(y)
            elif cons['type'] == 'ineq':
                if y < 0:
                    Constraints -= y

        for i in range(self.NObj):
            F[i] = -self.GP[i].predict(xx)[0] + Fator * Constraints

        return F

    # % __Sigmoid
    @staticmethod
    def __Sigmoid(x, k=10.):
        return 1./(1.+np.exp(k*(x-0.5)))

    def WriteSpace(self, filename="space"):

        Info = [self.space.NObj,
                self.space.NParam,
                self.space._NObs,
                self.space.length]

        np.savez(filename,
                 X=self.space._X,
                 Y=self.space._Y,
                 F=self.space._F,
                 I=Info)        # noqa

        return

    # % read relevant information from file

    def ReadSpace(self, filename="space.npz"):

        Data = np.load(filename)

        self.space.NObj = Data["I"][0]
        self.space.NParam = Data["I"][1]
        self.space._NObs = Data["I"][2]
        self.space.length = Data["I"][3]

        self.space._allocate((self.space.length + 1) * 2)

        self.space._X = Data["X"]
        self.space._Y = Data["Y"]
        self.space._F = Data["F"]

        # Redefine GP

        self.GP = [None] * self.NObj
        for i in range(self.NObj):
            self.GP[i] = GPR(kernel=Matern(nu=0.5),
                             n_restarts_optimizer=self.n_rest_opt)

        for i in range(self.NObj):
            yy = self.space.f[:, i]
            self.GP[i].fit(self.space.x, yy)

        self.__CalledInit = True
        self.N_init_points = self.space._NObs

        self.vprint("Read data from "+filename)

        return
