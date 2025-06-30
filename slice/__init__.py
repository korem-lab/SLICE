__version__ = "0.0.1"

from sklearn.utils import indexable, check_random_state#, metadata_routing
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples, check_array, column_or_1d
# from sklearn.utils.metadata_routing import _MetadataRequester
import numpy as np
import pandas as pd

# from abc import ABCMeta, abstractmethod

import numbers
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import type_of_target
from abc import ABCMeta, abstractmethod
from itertools import chain, combinations

def flatten(xss):
    return [x for xs in xss for x in xs]

# class GroupsConsumerMixin(_MetadataRequester):
#     """A Mixin to ``groups`` by default.

#     This Mixin makes the object to request ``groups`` by default as ``True``.

#     .. versionadded:: 1.3
#     """

#     __metadata_request__split = {"groups": True}
    
def get_site_vals(ytrain, ytest, grp_train, grp_test, site):
    train_mask = grp_train == site
    test_mask = grp_test == site
    tytr = ytrain[train_mask][ytrain[train_mask]!=-1]
    tyte = ytest[test_mask][ytest[test_mask]!=-1]
    return( pd.Series(tytr).value_counts(),
            pd.Series(tyte).value_counts())
    
    
def drop_overlapping_subgroups(ytrain, 
                               ytest, 
                               train_inds, 
                               test_inds, 
                               secondary_grptrain, 
                               secondary_grptest, 
                               min_n=5
                               ):
    


    overlapping_sites = set(secondary_grptrain[ytrain != -1]) & \
                        set(secondary_grptest[ytest != -1])

    if overlapping_sites != set():

        traincount = pd.Series( ytrain[ ytrain!=-1] ).value_counts()
        testcount = pd.Series( ytest[ ytest!=-1] ).value_counts()

        ## if we only have one type in the task/center, we can just mask the whole task
        if testcount.shape[0] == 1:
            ytest=-1

        if traincount.shape[0]==1:
            ytrain=-1

        if traincount.shape[0]==2 and testcount.shape[0]==2:


            traintotals = pd.Series( ytrain[ytrain!=-1] ).value_counts()
            testtotals = pd.Series( ytest[ytest!=-1] ).value_counts()

            ## figure out if we mask the train or the test site
            for site in pd.Series([a for a in overlapping_sites]).sort_values().values:
                #overlapping_sites: ## since the order looping through sets is random...
                site_train_counts, site_test_counts = get_site_vals(ytrain, 
                                                                    ytest, 
                                                                    secondary_grptrain,
                                                                    secondary_grptest,
                                                                    site)

                train_mask = secondary_grptrain == site
                test_mask = secondary_grptest == site

                possible_train_counts = traintotals.subtract(site_train_counts, fill_value=0)
                possible_test_counts = testtotals.subtract(site_test_counts, fill_value=0)

                 ## check how many datapoints we lose, mask labels as needed based on that
                if site_test_counts.sum()>site_train_counts.sum():
                    if possible_train_counts.min() >= min_n:
                        ytrain[train_mask] = -1
                        traintotals=possible_train_counts

                    else:
                        ytest[test_mask] = -1
                        testtotals=possible_test_counts
                else:
                    if possible_test_counts.min() >= min_n:
                        ytest[test_mask] = -1
                        testtotals=possible_test_counts
                    else:
                        ytrain[train_mask] = -1
                        traintotals=possible_train_counts
                    
    return(train_inds[ytrain!=-1], test_inds[ytest!=-1])

class SliceBaseCrossValidator(#_MetadataRequester, 
                              metaclass=ABCMeta):
    """Base class for all cross-validators.

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    
#     __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def split(self, X, y=None, groups=None, secondary_groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            
            train_index, test_index = \
                drop_overlapping_subgroups(y[train_index], 
                                           y[test_index], 
                                           train_index, 
                                           test_index, 
                                           secondary_groups[train_index], 
                                           secondary_groups[test_index], 
                                           min_n=5
                                           )

                        
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""

    def __repr__(self):
        return _build_repr(self)



class SliceOneGroupOut(#GroupsConsumerMixin, 
                       SliceBaseCrossValidator):
    """Leave One Group Out cross-validator.

    Provides train/test indices to split data such that each training set is
    comprised of all samples except ones belonging to one specific group.
    Arbitrary domain specific group information is provided as an array of integers
    that encodes the group of each sample.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    Read more in the :ref:`User Guide <leave_one_group_out>`.

    Notes
    -----
    Splits are ordered according to the index of the group left out. The first
    split has testing set consisting of the group whose index in `groups` is
    lowest, and so on.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneGroupOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 1, 2])
    >>> groups = np.array([1, 1, 2, 2])
    >>> logo = SliceOneGroupOut()
    >>> logo.get_n_splits(X, y, groups)
    2
    >>> logo.get_n_splits(groups=groups)  # 'groups' is always required
    2
    >>> print(logo)
    SliceOneGroupOut()
    >>> for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3], group=[2 2]
      Test:  index=[0 1], group=[1 1]
    Fold 1:
      Train: index=[0 1], group=[1 1]
      Test:  index=[2 3], group=[2 2]

    See also
    --------
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups
            )
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        return len(np.unique(groups))

    def split(self, X, y=None, groups=None, secondary_groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y.astype(int), groups, secondary_groups)

    
    
    
