import numpy as np
import math

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

        
def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)


def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))


def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def translate_sequence_to_states_old(sequence):
    N = len(sequence)
    states = np.array([])
    i = 0

    while i < N:
        nextS, lenA = checkStart(sequence[i: i + 3])
        states = np.append(states, nextS, axis = 0)
        i += lenA
        if states[-1] == 3 or states[-1] == 6 or states[-1] == 9:
            while states[-1] != 15 and states[-1] != 18 and states[-1] != 21:
                states = np.append(states, checkEndF(sequence[i : i + 3]), axis = 0)
                i += 3

        if states[-1] == 24 or states[-1] == 27 or states[-1] == 30:        
            while states[-1] != 36 and states[-1] != 39 and states[-1] != 42:
                states = np.append(states, checkEndR(sequence[i : i + 3]), axis = 0)
                i += 3

    return states


### TEST FOR HMM 7 ###
def createZ7(annotation):
    N = len(annotation)
    i = 0
    Z = np.zeros(N)

    while i < N:
        if i == 0:
            Z[i] = 3
            i += 1
        while annotation[i: i + 3] == 'CCC':
            Z[i: i + 3] = np.array([4, 5, 6])
            i += 3
        while annotation[i: i + 3] == 'RRR':
            Z[i: i + 3] = np.array([2, 1, 0])
            i += 3
        Z[i] = 3
        i += 1
        
    return Z


def createA(Z_list):
    A = np.zeros((43, 43))
    for Z in Z_list:
        for i in range(Z.shape[0] - 1):
            a, b = int(Z[i]), int(Z[i + 1])
            A[a, b] += 1

    for i in range(43):
        A[i] /= np.sum(A[i])

    return A


def createPi():
    Pi = np.zeros(43)
    Pi[0] = 1

    return Pi


def createPhi(Z_list, sequence_list):
    Phi = np.zeros((43, 4))
    for Z, s in zip(Z_list, sequence_list):
        for i in range(Z.shape[0]):
            state = int(Z[i])
            emission = int(s[i])
            Phi[state, emission] += 1

    for i in range(43):
        Phi[i] /= np.sum(Phi[i])

    return Phi

### END TEST FOR HMM 7 ###

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

def viterbi(A, Phi, Pi, sequence):
    N = len(sequence) # Number of steps in the markov chain
    K = 43 # Number of hidden states
    Omega = np.zeros((K, N))
    OmegaBack = np.zeros((K, N))

    # First column
    for i in range(K):
        Omega[i, 0] = log(Pi[i]) + log(Phi[i, sequence[0]])

    # Probably need log to make this work
    for i in range(1, N): # Loop over the sequence
        if i % 10000 == 0:
            print('{} viterbi\r'.format(i), end='')
        for k in range(K): # Loop over the hidden states
            Omega[k, i] = log(Phi[k, sequence[i]]) + np.max(Omega[:, i - 1] + np.log(A[:, k]))

    np.save('OmegaTest43.npy', Omega)
    # Backtracking
    Z = np.zeros(len(sequence))
    Z[-1] = np.argmax(Omega[:,-1])
    for i in reversed(range(0, N-1)):
        if i % 10000 == 0:
            print('{} backtracking\r'.format(i), end='')
        state = sequence[i+1]
        Z[i] = np.argmax(log(Phi[int(Z[i+1]), int(state)]) + Omega[:,i] + np.log(A[:, int(Z[i+1])]))
    return Z


def translate_sequence_to_states(sequence, annotation):
    N = len(sequence)
    states = np.zeros(N)
    i = 0
    while i < N:
        if (annotation[i-1: i + 3] == 'NCCC' or annotation[i-1: i + 3] == 'RCCC') and isStartF(sequence[i: i + 3]):
            states[i:i+3] = checkStart(sequence[i: i + 3])[0]
            i += 3
            while not annotation[i: i + 4] == 'CCCN':
                states[i:i+3] = np.array([10, 11, 12])
                i += 3
            states[i:i+3] = checkEndF(sequence[i : i + 3])
            i += 3
          
        if (annotation[i-1:i + 3] == 'NRRR' or annotation[i-1:i + 3] == 'CRRR') and isStartR(sequence[i:i+3]):
                states[i:i+3] = checkStart(sequence[i: i + 3])[0]
                i += 3
                while not annotation[i : i + 4] == 'RRRN':
                    states[i:i+3] = np.array([31, 32, 33])
                    i += 3
                states[i:i+3] = checkEndR(sequence[i : i + 3])
                i += 3
                
        if not annotation[i-1:i + 3] == 'RCCC':
            states[i] = 0
            i += 1
    return states

        
def isStartF(s):
    if s == 'ATG' or s == 'GTG' or s == 'TTG':
        return True
    else:
        return False

    
def isStartR(s):
    if s == 'TTA' or s == 'CTA' or s == 'TCA':
        return True
    else:
        return False

def isStopF(s):
    if s == 'TAG' or s == 'TGA' or s == 'TAA':
        return True
    else:
        return False

def isStopR(s):
    if s == 'CAT' or s == 'CAC' or s == 'CAA':
        return True
    else:
        return False


def checkStart(string):
    if string == 'ATG':
        return np.array([1, 2, 3]), 3

    if string == 'GTG':
        return np.array([4, 5, 6]), 3

    if string == 'TTG':
        return np.array([7, 8, 9]), 3

    if string == 'TTA':
        return np.array([22, 23, 24]), 3

    if string == 'CTA':
        return np.array([25, 26, 27]), 3

    if string == 'TCA':
        return np.array([28, 29, 30]), 3 

    return np.array([0]), 1


def checkEndF(string):
    if string == 'TAG':
        return np.array([13, 14, 15])

    if string == 'TGA':
        return np.array([16, 17, 18])

    if string == 'TAA':
        return np.array([19, 20, 21])

    return np.array([10, 11, 12])


def checkEndR(string):
    if string == 'CAT':
        return np.array([34, 35, 36])

    if string == 'CAC':
        return np.array([37, 38, 39])

    if string == 'CAA':
        return np.array([40, 41, 42])

    return np.array([31, 32, 33])


def calculateA(states):
    A = np.zeros((42, 42))
    for i in range(states.shape[0]-1):
        a, b = states[i], states[i + 1]
        A[a, b] += 1

    for i in range(42):
        A[i] /= np.sum(A[i])

    return A


def calculatePi():
    pi = np.zeros(42)
    pi[0] = 4
    pi[7] = 1

def convert_Z_to_ann7(Z):
    ann = ''
    for i in range(len(Z)):
        if Z[i] == 3:
            ann += 'N'
        elif Z[i] > 3 :
            ann += 'C'
        elif Z[i] < 3 :
            ann += 'R'            
        
    return ann

def convert_Z_to_ann(Z):
    ann = ''
    for i in range(len(Z)):
        if Z[i] == 0:
            ann += 'N'
        elif 1 <= Z[i] <= 21 :
            ann += 'C'
        elif 22 <= Z[i] <= 42 :
            ann += 'R'            
        
    return ann

genomes = {}
annotation = {}
Z = [None]*5
sequence_list = [None]*5
for i in range(1, 6):
    sequence = read_fasta_file('genome' + str(i) + '.fa')
    sequence_list[i - 1] = translate_observations_to_indices(sequence['genome' + str(i)])
    genomes['genome' + str(i)] = sequence['genome' + str(i)]
    ann = read_fasta_file('true-ann' + str(i) + '.fa')
    annotation['genome' + str(i)] = ann['true-ann' + str(i)]
    # Test for hmm7
    Z[i-1] = translate_sequence_to_states(genomes['genome' + str(i)], annotation['genome' + str(i)])
    # Z[i-1] = createZ7(annotation['genome' + str(i)])
    #print(Z[i-1][-10:])

    
A = createA(Z[:4])
Phi = createPhi(Z[:4], sequence_list[:4])
Pi = createPi()


sequence = sequence_list[4]

#print('Transition probabilities are', A)
#print('Emission probabilities are', Phi)
Zml = viterbi(A, Phi, Pi, sequence)
#print(Zml[-100:])
np.save('Z_5.npy', Zml)

#states = translate_sequence_to_states(genomes['genome1'])
#np.save('genome1.npy', states)

#Omega = np.load('OmegaTest2.npy')
#print(Omega[:,-10:].T)

#Z2_7 = np.load('Ztest2_7.npy')
ann = convert_Z_to_ann(Zml)
file = open("pred-ann5.fa", "w")
file.write(ann)
file.close()
