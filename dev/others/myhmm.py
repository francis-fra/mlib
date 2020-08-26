# dervid from Rabina tutorial
from string import ascii_lowercase
import random
import numpy as np

class HMM(object):
    
    def __init__(self, A, B, T, vocab=None, p=None):
        self.A = A
        self.B = B
        self.T = T
        # M: num of states
        # N: num of observable vocabulary
        M, N = B.shape
        self.M = M
        self.N = N
        if vocab is None:
            self.vocab = ascii_lowercase[:N]
        else:
            self.vocab = vocab[:N]
            
        if p is None:
            self.p = [1.0/M] * M
        else:
            self.p = p
        
        
    def get_num_states(self):
        return (self.M)
    
    def get_num_vocab(self):
        return (self.N)
    
    def get_seq_length(self):
        return (self.T)
        
    def get_sequence(self, state=None):
        "generate sequence and hidden states"
        
        # generate initial state (in time 0)
        if state is None:
            state = self.get_state_by_random(self.p)
            
        outSequence = np.zeros(self.T)
        outState = np.zeros(self.T)
        
        for k in range(self.T):
            outState[k] = state
            # gen next state and current output
            state, out = self.gen_output(state)
            outSequence[k] = out

        return (outSequence, outState)
        
    def gen_output(self, state):
        "get output and state at one time"
        
        new_state = self.get_state_by_random(self.A[state])
        out = self.get_state_by_random(self.B[state])
        
        return (new_state, out)
        
            
    def get_state_by_random(self, p): 
        "get state from probability distribution"
        
        prob = np.cumsum(p)
        # uniform distribution
        r = random.random()
        S0 = min(np.where(prob >= r)[0])
        return S0            
            
            
    def forward_probability(self, outSequence, S0=None):
        "calculate probability of the supplied sequence"
        
        # probability of the partial observation sequence up to time t
        # at the current state
        
        # initialization
        alpha_hist = np.zeros([self.M, self.T])
        out = outSequence[0]
        
        if S0 is None:
            for k in range(self.M):
                alpha_hist[k,0] = self.p[k] * self.B[k][out]
        else:
            alpha_hist[S0,0] = 1
            
        # induction
        for t in range(1, self.T):
            out = outSequence[t]
            for k in range(self.M):
                # propagation probability to state k from other states at t-1
                alpha_hist[k,t] = np.dot(alpha_hist[:,t-1], self.A[:,k])
                # probability multiplied by the prob of observing symbol out
                alpha_hist[k,t] = alpha_hist[k,t] * self.B[k][out]
     
        return alpha_hist
    
    def backward_probablitity(self, outSequence):
        "backward calculation"

        # probability of the partial observation sequence from time t+1
        # to the end at the current state
                
        # initialization
        beta_hist = np.zeros([self.M, self.T])
        
        for k in range(self.M):
            beta_hist[k, -1] = 1
            
        # induction
        for t in range(self.T-2, -1, -1):
            out = outSequence[t+1]
            for k in range(self.M):
                # elementwise multiply beta with output probability first
                prob = np.multiply(beta_hist[:, t+1], self.B[:, out])
                # dot product of the probability
                beta_hist[k, t] = np.dot(prob, self.A[k,:])
        
        return beta_hist
    
    def decoding(self, outSequence):
        "Viterbi algorithm: find the most likely hidden states"
        
        # initialization
        # it stores the probability at each state at each time
        gamma_hist = np.zeros([self.M, self.T])
        # it stores the most likely node propagated from previous time step
        node_hist = np.zeros([self.M, self.T])
        out = outSequence[0]
        
        for k in range(self.M):
            gamma_hist[k, 0] = self.p[k] * self.B[k][out]
                      
                
        # recursion
        for t in range(1, self.T):
            out = outSequence[t]
            for k in range(self.M):
                # find the propagation probability for each potential node from time t-1
                candidates = np.multiply(gamma_hist[:,t-1], self.A[:,k]) * self.B[k][out]
                # max probability
                gamma_hist[k, t] = max(candidates)
                # hidden node index (the most likely node propagated from time t-1) 
                node_hist[k, t] = np.argmax(candidates)
                
        # termination states
        P = max(gamma_hist[:,-1])
        Q = np.argmax(gamma_hist[:,-1])

        # most likely decoded path
        ml_path = np.zeros(self.T, dtype=np.int32)
        ml_path[-1] = Q
        
        # back tracking
        for t in range(self.T-2, -1, -1):
            ml_path[t] = node_hist[ml_path[t+1], t+1]

#         return (gamma_hist, node_hist, ml_path)
        return ml_path
    
    
    def forward_backward_probability(self, outSequence):
        "calculate forward backward recursion"
        
        alpha_hist = self.forward_probability(outSequence)
        beta_hist = self.backward_probablitity(outSequence)
        
        gamma_hist = np.zeros([self.M, self.T])
        
        for t in range(self.T):
            normalized_factor = np.dot(alpha_hist[:,t], beta_hist[:, t])
            gamma_hist[:,t] = np.multiply(alpha_hist[:,t], beta_hist[:,t]) / normalized_factor
            
        return gamma_hist
    
    def decode_sequence(self, outSequence):
        "decode sequence using forward-backward algorithm"
        
        gamma_hist = self.forward_backward_probability(outSequence)
        ml_path = np.zeros(self.T, dtype=np.int32)
        for t in range(self.T):
            ml_path[t] = np.argmax(gamma_hist[:,t])
        
        return ml_path
    
    def evaluate(self, numTrials, method="viterbi", print=False):
        "calculate accuracy of decoding"
        
        if method == 'viterbi':
            f = self.decoding
        else:
            f = self.decode_sequence
    
        count_right = 0.0
        for k in range(numTrials):
            outseq, outState = self.get_sequence()
            outseq = outseq.astype(int)
            decoded_path = f(outseq)
            if np.allclose(outState, decoded_path):
                count_right += 1
        
        if print:
            print("Accuracy {}".format(count_right / numTrials))
            
        return (count_right / numTrials)
    
    def learning(self, outSequence):
        "learn HMM parameters"
        
        eta_hist = np.zeros([self.T, self.M, self.M])
        
        beta_hist = self.backward_probablitity(outSequence)
        alpha_hist = self.forward_probablitity(outSequence)
        
        for t in range(self.T-1):
            out = outSequence[t]
            next_out = outSequence[t+1]
            normalized_factor = 0
            for k in range(self.M):
                for j in range(self.M):
                    value = alpha_hist[k,t] * self.A[k,j] * self.B[j,next_out] * beta_hist[j, t+1]
                    normalized_factor += value
                    eta_hist[t,k,j] = value
                    
            # normalize
            eta_hist[t,:,:] = eta_hist[t,:,:] / normalized_factor
            
        # TODO: find estimated alpha and beta prbabilities
        
        return eta_hist
    
    
        