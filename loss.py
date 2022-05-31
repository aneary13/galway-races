import numpy as np
from scipy import special


# Define a class for focal loss
class FocalLoss:

    # Pass values for alpha and gamma
    def __init__(self, gamma=2, alpha=0.25):
        self.alpha = alpha
        self.gamma = gamma

    # Calculate alpha_t
    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    # Calculate p_t
    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    # Define the focal loss
    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    # Define the first order derivative
    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    # Define the second order derivative
    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    # Objective function to be used during training (focal loss)
    def f_obj(self, pred, train_data):
        y = train_data.get_label()
        p = special.expit(pred)
        return self.grad(y, p), self.hess(y, p)

    # Evaluation function to be used during training (focal loss)
    def f_eval(self, pred, train_data):
        y = train_data.get_label()
        p = special.expit(pred)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better
