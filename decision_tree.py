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



