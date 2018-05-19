# decision-tree  
Decision tree implementation using Information Gain Heuristic and Variance Impurity Heuristic.  
Download datasets 1 and 2.

# commands  
Compile : javac com/utd/ml/dtree/ID3.java  
Run: java com.utd.ml.dtree.ID3 <L> <K> <training-set> <validation-set> <test-set> <to-print>  
where L and K are positive integer values used in post-pruning algorithm, to-print: print the decision tree or not (yes, no)  
training-set, validation-set and test-set are paths to training dataset, validation dataset and test dataset respectively.  

Test arguments: 3 4 \data_sets1\training_set.csv \data_sets1\validation_set.csv \data_sets1\test_set.csv yes
