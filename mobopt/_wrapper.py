from sklearn.gaussian_process import GaussianProcessRegressor
# from smt.surrogate_models import KPLS


class GaussianProcessWrapper(GaussianProcessRegressor):
    pass


# class GaussianProcessWrapper:
#     def __init__(self, *args, **kwargs):
#         self.sm = KPLS(theta0=[1e-2], print_global=False)
#         print("KPLS method")

#     def fit(self, X, y):
#         self.sm.set_training_values(X, y)
#         self.sm.train()

#     def predict(self, X):
#         return self.sm.predict_values(X).ravel()
