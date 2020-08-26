import numpy as np

srcDir = '/home/fra/FraDir/learn/Learnpy/Mypy/hmm'
import sys
sys.path.append(srcDir)
from myhmm import HMM

from imp import reload

#----------------------------------------------------------------------
# State transition matrix
A = np.array([
             [1.0, 0.0, 0.0, 0.0],
             [0.2, 0.3, 0.1, 0.4],
             [0.2, 0.5, 0.2, 0.1],
             [0.8, 0.1, 0.0, 0.1]
             ])

# observation generation matrix
B = np.array([
             [1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.3, 0.4, 0.1, 0.2],
             [0.0, 0.1, 0.1, 0.7, 0.1],
             [0.0, 0.5, 0.2, 0.1, 0.2]
             ])

# length of sequence
T = 5

outseq = [3, 1, 3, 2, 0]

#----------------------------------------------------------------------
A = np.array([
             [0.4, 0.3, 0.3],
             [0.2, 0.6, 0.2],
             [0.1, 0.1, 0.8]
             ])

# observation generation matrix
B = np.array([
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]
             ])

vocab = ['rain', 'cloudy', 'sunny']

outseq = [0, 0, 2, 2, 1]

#----------------------------------------------------------------------
# initital state
S0 = 2

import myhmm
reload(myhmm)
from myhmm import HMM

# hmm = HMM(A, B, T)
# hmm.get_sequence(1)
# hmm = HMM(A, B, T, vocab)
# hmm.get_sequence()

#----------------------------------------------------------------------
# find probability

hmm = HMM(A, B, T=5)
outseq
hmm.forward_probability(outseq)
# tmp = hmm.forward_probability(outseq, 1)
# tmp

hmm.backward_probablitity(outseq)
hmm.forward_backward_probability(outseq)
hmm.decode_sequence(outseq)

gamma, node = hmm.decoding(outseq)
gamma
node
hmm.decoding(outseq)

#----------------------------------------------------------------------
# generate sequence
#----------------------------------------------------------------------
hmm.get_sequence()
outseq, outState = hmm.get_sequence()
outseq, outState 
outseq
# array([ 2.,  4.,  1.,  1.,  0.])
outState
# array([ 1.,  2.,  2.,  3.,  0.])

# outseq = np.array([ 2.,  2.,  0.,  0.,  0.])
outseq = outseq.astype(int)
outseq

#----------------------------------------------------------------------
# backtracking
import myhmm
reload(myhmm)
from myhmm import HMM
hmm = HMM(A, B, T=5)
hmm.decoding(outseq)

hmm.evaluate(100)
hmm.evaluate(100, "other")

np.mean([hmm.evaluate(100) for k in range(100)])
np.mean([hmm.evaluate(100, "other") for k in range(100)])
    


#----------------------------------------------------------------------
# DEBUG
#----------------------------------------------------------------------
a = np.array([1, 1, 1, 3, 0], dtype=np.int32)
a
b = np.array([1, 1, 1, 3, 0], dtype=np.int32)
np.allclose(a, b)


A[1]
A
A[:,1]
hmm.p
hmm.vocab
B[1][2]

tmp = np.zeros([3,5])
tmp[:,1] = hmm.p
tmp
a = np.array([1,3,4])
b = np.array([2,-3,-4])

np.multiply(a, b)

list(range(6, -1, -1))