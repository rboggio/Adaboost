from typing import Optional
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import (check_X_y, check_array, check_is_fitted)


class AdaBoost(BaseEstimator, ClassifierMixin):
    """
    A custom implementation of the discrete AdaBoost (SAMME) algorithm with
    DecisionTreeClassifier as the weak learner. This version accommodates
    arbitrary binary labels by using LabelEncoder internally.

    Parameters
    ----------
    n_estimators : int, default=5
        The maximum number of estimators at which boosting is terminated.
        In the majority of cases, boosting terminates early if it perfectly
        fits the training data.
    random_state : int or None, default=None
        Controls the random seed given to each DecisionTreeClassifier.
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(self, n_estimators: int = 5, random_state: Optional[int] = None) -> None:
        
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "AdaBoost":
        """
        Build a boosted classifier from the training set (X, y).

        Internally, the labels y are encoded as {0, 1} via a LabelEncoder.
        Each weak learner (decision stump) is trained on these encoded labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            The target values. Must be only two classes.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate X, y
        X, y = check_X_y(X, y)

        # Encode labels as {0, 1} for the decision stumps
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        if len(self._le.classes_) != 2:
            raise ValueError("AdaBoost requires exactly 2 distinct classes in y.")

        n_samples = X.shape[0]

        # Initialize weights for all samples equally
        self.weights_ = np.ones(n_samples) / n_samples

        # Containers for weak learners and their alphas
        self.models_ = []
        self.alphas_ = []

        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            stump.fit(X, y_enc, sample_weight=self.weights_)

            # Stump predictions in the encoded space {0, 1}
            stump_pred_enc = stump.predict(X)

            # Convert predictions back to original labels (e.g., {-1, +1})
            stump_pred = self._le.inverse_transform(stump_pred_enc)

            # Calculate weighted error
            misclassified = (stump_pred != y)
            error = np.sum(self.weights_ * misclassified)

            # Compute alpha for discrete AdaBoost (SAMME)
            alpha = 0.5 * np.log((1 - error) / error + 1e-12)

            # Convert original y to numeric form {-1, +1} for weight update
            numeric_y = np.where(y == self._le.classes_[0], -1.0, 1.0)
            numeric_pred = np.where(stump_pred == self._le.classes_[0], -1.0, 1.0)

            # Update sample weights
            self.weights_ *= np.exp(-alpha * numeric_y * numeric_pred)
            self.weights_ /= np.sum(self.weights_)

            # Store the trained stump and alpha
            self.models_.append(stump)
            self.alphas_.append(alpha)

        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Predict class labels for the given input samples X.

        The prediction is made by taking a weighted (via alpha) majority vote
        over all weak learners.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels, in the original label space.
        """
        check_is_fitted(self, ["models_", "alphas_"])
        X = check_array(X)

        # Aggregate predictions in numeric space {-1, +1}
        agg = np.zeros(X.shape[0], dtype=float)
        for stump, alpha in zip(self.models_, self.alphas_):
            stump_pred_enc = stump.predict(X)  # predictions in [0, 1]
            stump_pred = self._le.inverse_transform(stump_pred_enc)  # back to original
            numeric_pred = np.where(stump_pred == self._le.classes_[0], -1.0, 1.0)
            agg += alpha * numeric_pred

        # Convert sign(agg) to numeric labels: -1.0 or +1.0
        numeric_final = np.where(agg >= 0, 1.0, -1.0)

        # Decode numeric_final back to original classes
        final_pred = np.empty_like(numeric_final, dtype=self._le.classes_.dtype)
        mask_pos = (numeric_final == 1.0)
        final_pred[~mask_pos] = self._le.classes_[0]  # -1.0 side
        final_pred[mask_pos] = self._le.classes_[1]   # +1.0 side

        return final_pred
