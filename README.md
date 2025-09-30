**SLICE**
=============
<!-- badges: start -->
<!-- [![main](https://github.com/korem-lab/SLICE/actions/workflows/main.yml/badge.svg)](https://github.com/korem-lab/SLICE/actions/workflows/main.yml)
<!-- badges: end -->
<img src='vignettes/SLICE_logo.png' align="right" height="250" />

### SLICE - Site-Level Independent Cross-Evaluation

This is a python package designed to facilitate conservative cross-validations schemes that account for complex multi-level experimental designs in computational biology. To begin using SLICE, we recommend reading this README and implementing the code demonstration below.
<!--  reading it's [documentation pages](https://korem-lab.github.io/SLICE/). -->


All classes from this package provide train/test indices to split data in train/test sets to account for potential multi-layer batch biases. This package is designed to enable automated cross-validation in a format similar to scikit-learn's `LeaveOneGroupOut`. 
For any support using SLICE, please use our <a href="https://github.com/korem-lab/SLICE/issues">issues page</a> or email: gia2105@columbia.edu.

**Installation**
-------------------
```bash
pip install git+https://github.com/korem-lab/SLICE.git
```
The dependencies for SLICE are python, numpy, and scikit-learn. Is has been developed and tested using python 3.6 - 3.12. Only standard hardware is required for SLICE. The typical install time for SLICE is less that 15 seconds. 


**Example**
-----------------
We demonstrate the following snippet of code to utilize SLICE, using an observation matrix `X`, a binary outcome vector `y`, and two batch groupings `b1` and `b2`. We demonstrate it using scikit-learn's `LogisticRegressionCV`, although this can be replaced with any training/tuning/predicting scheme. The expected runtime for the SLICE package's operations are less than 5 seconds, while the overall example is expected to complete in less than one minute on most machines.

```python
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from slice import SliceOneGroupOut
from sklearn.metrics import roc_auc_score
np.random.seed(1)

## given some random `X` matrix, and a `y` binary vector
X = np.random.rand(100, 10)
y = np.random.rand(100) > 0.5

## Leave-one-out evaluation
loo = LeaveOneOut()
loocv_predictions = [ LogisticRegressionCV()\
                                .fit(X[train_index], y[train_index])\
                                .predict_proba(X[test_index]
                                            )[:, 1][0]
              for train_index, test_index in loo.split(X, y) ]

## Since all the data is random, a fair evaluation
## should yield au auROC close to 0.5
print('Leave One Out auROC: {:.2f}'\
              .format( roc_auc_score(y, loocv_predictions) ) )

## Rebalanced leave-one-out evaluation
rloo = RebalancedLeaveOneOut()
rloocv_predictions = [ LogisticRegressionCV()\
                                .fit(X[train_index], y[train_index])\
                                .predict_proba(X[test_index]
                                            )[:, 1][0]
              for train_index, test_index in rloo.split(X, y) ]

## Since all the data is random, a fair evaluation
## should yield au auROC close to 0.5
print('Rebalanceed Leave-one-out auROC: {:.2f}'\
              .format(  roc_auc_score(y, rloocv_predictions) ) )
```

    Leave One Out auROC: 0.00
    Rebalanceed Leave-one-out auROC: 0.48


As demontrated in this example, neglecting to account for inner batch structures cann introduce biases in evaluations.



**Classes**
---------

### SliceOneGroupOut

Provides train/test indices to split data in train/test sets with rebalancing to ensure that all training folds have identical class balances. Each sample is used once as a test set, while the remaining samples form the training set. See sklearn.model_selection.LeaveOneOut for more details on Leave-one-out cross-validation. 

##### **Parameters**
No parameters are used for this class 

### RebalancedKFold

Provides train/test indices to split data in `n_splits` folds, with rebalancing to ensure that all training folds have identical class balances. Each sample is only ever used within a single test fold. `RebalancedKFold` uses the following parameters, which are the same as the scikit-learn `StratifiedKFold` parameters (see sklearn.model_selection.StratifiedKFold for more details):

##### **Parameters**
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.



**Citation**
-------
TBD
Austin, G.I. et al. “Conservative evaluation demonstrates tumor-specific microbial signatures that generalize across processing pipelines” (2025). LINK TBD
