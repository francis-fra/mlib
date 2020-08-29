# TODO: refactor multiclass and binary classification evaluators

#---------------------------------------------------------------------
# classifiers
#---------------------------------------------------------------------
class RandomClassifier(BaseEstimator):
    """
        Parameters
        ----------
        prop : base rate
    """
    def __init__(self, prop):
        self.prop = prop
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        nums = np.zeros((len(X), 1), dtype=bool)
        N = math.floor(len(X) * self.prop)
        nums[:N] = True
        np.random.shuffle(nums)
        return nums

class NeverClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)