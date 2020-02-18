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
    import MOBOpt.bayes as by
    import deap.benchmarks as db

    def target(x):
        return - np.asarray(db.zdt1(x)) <!-- The method finds the max of the target function -->

    ```
