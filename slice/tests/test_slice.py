#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from slice import SliceOneGroupOut
from sklearn.metrics import roc_auc_score
import unittest


class SLICEtest(unittest.TestCase):
    def run_classification_cv(self, 
                              cv_object, 
                              seed=0):


        np.random.seed(seed)
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

        
        self.assertTrue( mean_auroc > 0.4 and mean_auroc<0.6 )
        
        for train_index, test_index in sogo.split(X, 
                                                  y, 
                                                  groups=b2, 
                                                  secondary_groups=b1):
            unique_train_b1 = np.unique(b1[train_index])
            unique_test_b1 = np.unique(b1[test])
            for a in unique_train_b1:
                self.assertTrue(a not in unique_test_b1)


    
if __name__ == '__main__':
    unittest.main()