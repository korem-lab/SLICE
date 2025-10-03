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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneGroupOut
from slice import SliceOneGroupOut
from sklearn.metrics import roc_auc_score

np.random.seed(0)
n_samples=750
n_features=25
n_b1 = 7
n_b2 = 3

## given some random `X` matrix, and two categorical batch groupings
X = np.random.normal(size=(n_samples, n_features))
b1 = np.random.randint(low=0, high=n_b1, size=n_samples)
b2 = np.random.randint(low=0, high=n_b2, size=n_samples)

## simulate batch biases for X and label y, for simplicity we'll just add biased noise to X
b1_biases = np.random.normal(scale=0.25, size=(n_b1, n_features))
b2_biases = np.random.normal(scale=0.25, size=(n_b2, n_features))
X += b1_biases[b1] + b2_biases[b2]

## generate y labels, only biased by batches, no direct relationship with X
##     Therefore, by construction, an unbiased auROC is approx. 0.5
b1_label_biases = np.random.normal(loc=b1, scale=.7)
b2_label_biases = 0* np.random.normal(loc=b2, scale=5)
y = ( b1_label_biases + b2_label_biases ) > 0

## Run Cross-validations

## Leave-one-group-out auROCs
logo=LeaveOneGroupOut()
logo_aurocs = [ roc_auc_score(y[test_index],
                              LogisticRegressionCV(solver='newton-cg')\
                                        .fit(X[train_index], y[train_index])\
                                        .predict_proba(X[test_index]
                                                    )[:, 1]
                             )
              for train_index, test_index in logo.split(X, y, b2) ]

## SLICE-one-group-out auROCs
sogo=SliceOneGroupOut()
sogo_aurocs = [ roc_auc_score(y[test_index],
                              LogisticRegressionCV(solver='newton-cg')\
                                        .fit(X[train_index], 
                                             y[train_index])\
                                        .predict_proba(X[test_index]
                                                    )[:, 1]
                             )
              for train_index, test_index in sogo.split(X, 
                                                        y, 
                                                        groups=b2, 
                                                        secondary_groups=b1) ]

## the unaccounted bias from an inner batch grouping produced erronesouly high auROCs
##      even in a standard 'Leave-one-batch-out' approach
print("Leave-one-group-out auROCs:", *map("{:.2f}".format, logo_aurocs))

## SLICE resolved these biases, producing the correct null performance
##      by accounting for both batch leyers during cross-validation
print("SLICE-one-group-out auROCs:", *map("{:.2f}".format, sogo_aurocs))
```

    Leave-one-group-out auROCs: 0.64 0.73 0.71
    SLICE-one-group-out auROCs: 0.44 0.50 0.46


As demontrated in this example, neglecting to account for inner batch structures can introduce biases in evaluations.



**Classes**
---------
The main class in the SLICE package is `SliceOneGroupOut`. 

### SliceOneGroupOut

    Provides train/test indices to split data such that each training set is
    comprised of all samples except ones belonging to one specific group, while
    ensuring that any secondary group cannot be present in both train and test sets.
    Arbitrary domain specific group information is provided as an array of integers
    that encodes the group of each sample.

    For instance the groups could be the etraction batch of collection of the samples, 
    centers from which samples were processed, or others. 

    Designed to have similar functionality as scikit-learn's `LeaveOneGroupOut`

The main method used in by `SliceOneGroupOut` is `.split`:

#### split(self, X, y=None, groups=None, secondary_groups=None, min_n_per_class=5):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels used to separate train and test sets
            while splitting the dataset.
            
        secondary_groups : array-like of shape (n_samples,), default=None
            Secondary group labels for the samples used for selective 
            subsampling to ensure no secondary group is present in both train and test sets.
            
        min_n_per_class : integer, default=5
            Determines the minimum number of samples per class that SLICE tries to preserve
            during stratification and sample removal.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """


**Citation**
-------
Austin, G.I. et al. “Tumor-specific microbial signatures generalize across clinical sites, laboratories, and bioinformatic pipelines” (2025).
