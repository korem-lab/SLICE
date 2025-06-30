import numpy as np
import pandas as pd
from slice import SliceOneGroupOut
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([1, 2, 1, 2])
# groups = np.array([1, 1, 2, 2])
# sogo = SliceOneGroupOut()
# sogo.get_n_splits(X, y, groups)
# sogo.get_n_splits(groups=groups)
# for i, (train_index, test_index) in enumerate(sogo.split(X, y, groups)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}, group={groups[train_index]}")
#     print(f"  Test:  index={test_index}, group={groups[test_index]}")



X = np.random.rand(250, 20)
y = np.random.rand(250) > 0.8
groups = ( np.random.rand(250) * 5 ).round()
secondary_groups = ( np.random.rand(250) * 10 ).round()

sogo = SliceOneGroupOut()
for i, (train_index, test_index) in enumerate(sogo.split(X, 
                                                         y, 
                                                         groups, 
                                                         secondary_groups)):
    print(f"Fold {i}:")
#     print(np.unique(secondary_groups[train_index]))
#     print(np.unique(secondary_groups[test_index]))

    print( pd.Series(np.unique(secondary_groups[train_index]))\
                      .isin( np.unique(secondary_groups[test_index]) ))
    
    
#     print(np.unique(secondary_groups[test_index]) in \
#           np.unique(secondary_groups[train_index]) )
    