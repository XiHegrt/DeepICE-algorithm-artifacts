import numpy as np
from coreset import Deep_ICE_coreset
from Deep_ICE import Deep_ICE

# Load the CSV file
data = np.loadtxt("datasets/voicepath_data.csv", delimiter=",")

data = np.unique(data, axis=0)

X = data[:,:-1]
t = data[:,-1]
N,D = X .shape

X = np.unique(X, axis=0)

min_val = X.min(axis=0)   # Minimum value of each column
max_val = X.max(axis=0)   # Maximum value of each column

epsilon = 1e-8
X = 2 * (X - min_val) / (max_val - min_val+epsilon) - 1

# Add noise to the dataset
np.random.seed(2024)
noise_std_dev = 1e-8
noise = np.random.normal(0, noise_std_dev, size=X.shape)
X_noisy = X + noise

### deep-ICE with coreset selection

K=2 #number of hyperplanes
L = 10 #heap size
M = 20 #block size
max_unchanged = 1 #count the number of unchange of coreset
Bmax = 25 #the maximal block size that deep-ice algorithm can process
threshold = 5 #the threshold for unchange
num_candidates=500 #number of candidate configurations


best_candidates= Deep_ICE_coreset(X, t, K, L, M, max_unchanged, Bmax, threshold, num_candidates=500)
best_candidates = [(vec, val.cpu().item(),block) for vec, val,block in best_candidates]

print(f'The best rank-{K} maxout network with a single maxout neuron, obtained using the Deep-ICE coreset, achieves a 0-1 loss of {best_candidates[1]}. It is constructed via the nested combination {best_candidates[0]}, and the corresponding block is {best_candidates[2]}')

### complete run of Deep-ICE algorithm

inds = [i for i in range(N)]
inds = inds[:132] # python version can not process the whole dataset, but the exact solution will be find at stage.
res = Deep_ICE(inds[:132], X, t, K, 500, P=True)

print(f'The optimal rank-{K} maxout network with one maxout neuron has a 0-1 loss {res[1]}, which is constructed by nested combination {res[0]}')