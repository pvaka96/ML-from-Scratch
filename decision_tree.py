import numpy as np

def entropy(y):
    hist = np.bincount(y)
    # Does this bincount() work only if y has positive integers. Verify through documentation?
    '''
    ps store a number of probabilities for all number of occurences.
    '''
    ps = hist / len(y)
    '''
    We consider applying p*log2(p) only if p > 0. so if a bin value is zero, then we wont get any math error
    '''
    return -np.sum([p*np.log2(p) for p in ps if p > 0])

class Node:
    
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self. feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        # is not None returns a bool value.
        return self.value is not None

class DecisionTree:
    
    def __init__(self, min_sample_split=2, max_depth=100, n_feats=None):
        
        '''
        min_sample_split = minimum samples required to further split the tree
        max_depth = maximum depth the tree can go to
        n_feat = We do a greedy serach over the features but we loop over a subset of features only.
                Feature subset is randomly selected. This variable is like one of those random factors. 
                This is also one of the reasons why it is called a Random Forest.
        '''
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
    
    def fit(self, X, y):
        # Grow Tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X,y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
    def predict(self, X):
        # Traverse Tree
        