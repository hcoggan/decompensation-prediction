import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from plotnine import *
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin


def load_existing_models(path):
    """Load a previously-saved fitted-models.joblib, if present. Returns an
    empty dict if it doesn't exist yet or can't be read (e.g. first run)."""
    if not os.path.exists(path):
        return {}
    try:
        existing = joblib.load(path)
        if not isinstance(existing, dict):
            return {}
        return existing
    except Exception as e:
        print(f"Could not load existing fitted models at {path} ({e}); refitting all models.")
        return {}
    


#----------AGE-IMPUTATION--------

age_bins = [
    -np.inf, 91.25, 182.5, 365, 547.5, 1095, 1825, 3650, 5475, np.inf
]
age_labels = [
    "0-3 mo", "3-6 mo", "6-12 mo", "12-18 mo", "18-36 mo",
    "3-5 y", "5-10 y", "10-15 y", "15+ y"
]

class AgeAwareMeanImputer(BaseEstimator, TransformerMixin):
    """
    Mean-imputes NaNs in a numeric feature matrix, for use ahead of the MLP
    (which can't otherwise handle missing values).

    - For vitals columns whose name contains "hr", "rr", or "weight"
      (case-insensitive substring match), missing values are filled with the
      AGE-GROUP-specific mean, computed on the training data only, using the
      age groups defined by `age_bins`/`age_labels` and the `age_col` column.
    - For every other column (including `age_col` itself), missing values
      are filled with the overall training-set mean.

    All means are learned in `fit` (i.e. from the training fold only) and
    simply applied in `transform`, so no test-set information leaks into
    the imputation.
    """

    VITALS_KEYWORDS = ("hr", "rr", "weight")

    def __init__(self, age_col='age_in_days'):
        self.age_col = age_col

    def _is_vitals_col(self, col):
        c = str(col).lower()
        return any(k in c for k in self.VITALS_KEYWORDS)

    def _age_groups(self, X):
        return pd.cut(X[self.age_col], bins=age_bins, labels=age_labels)

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        if self.age_col not in X.columns:
            raise ValueError(
                f"AgeAwareMeanImputer requires an '{self.age_col}' column in X "
                "to compute age groups."
            )

        self.columns_ = list(X.columns)
        self.vitals_cols_ = [
            c for c in self.columns_ if c != self.age_col and self._is_vitals_col(c)
        ]
        self.other_cols_ = [
            c for c in self.columns_ if c not in self.vitals_cols_ and c != self.age_col
        ]

        age_groups = self._age_groups(X)

        # Age-group-specific means for vitals columns, learned on the training data.
        self.age_group_means_ = {
            c: X.groupby(age_groups)[c].mean() for c in self.vitals_cols_
        }
        # Overall training means as a fallback, in case some age group has no
        # non-missing observations for a given vitals column in the training fold.
        self.overall_vitals_means_ = X[self.vitals_cols_].mean()

        # Overall training means for every other column, including age_col itself.
        self.overall_means_ = X[self.other_cols_ + [self.age_col]].mean()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        # pd.cut() returns a Categorical Series; .map() on a Categorical stays
        # Categorical too, and then fails when we try to fillna() it with a
        # value (the overall mean) that isn't one of its existing categories.
        # Casting to str first forces a plain, non-categorical result.
        age_groups = self._age_groups(X).astype(str)

        for c in self.vitals_cols_:
            group_fill = age_groups.map(self.age_group_means_[c])
            group_fill = group_fill.astype(float).fillna(self.overall_vitals_means_[c])
            X[c] = X[c].fillna(group_fill)

        for c in self.other_cols_ + [self.age_col]:
            X[c] = X[c].fillna(self.overall_means_[c])

        return X

