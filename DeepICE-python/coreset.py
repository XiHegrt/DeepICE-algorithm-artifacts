import random
import bisect
from auxfuncs_array import *
from Deep_ICE import Deep_ICE
import time



# add a configuration to a heap
def add_cnfg(cnfgs, new_config, size):

    # Extract the objective values for the sorting process
    objective_values = [cnfg[1] for cnfg in cnfgs]
    
    # If the list has fewer than 10 items or the new config has a better (smaller) objective value
    if len(cnfgs) < size or new_config[1] < cnfgs[-1][1]:
        # Find the position to insert based on the objective value
        idx = bisect.bisect_left(objective_values, new_config[1])
        
        # Insert the new configuration at the correct position
        cnfgs.insert(idx, new_config)
        # print(f'the container has a new element {new_config}')
         
        # If the list exceeds size 10, remove the configuration with the largest objective value
        if len(cnfgs) > size:
            cnfgs.pop(-1)
    
    return cnfgs


def Deep_ICE_coreset(X, t, K, L, M, max_unchanged, Bmax, threshold, num_candidates=500):
    """coreset selection method

    Args:
        X (N,D): input data
        t (N,): label vector, consists of -1 and 1
        K (int): number of hyperplane
        L (int): heap size
        M (int): block size
        max_unchanged (int): count the number of unchange of coreset
        Bmax (int): the maximal block size that deep-ice algorithm can process
        threshold (int): the threshold for unchange
        num_candidates (int, optional): number of candidate configurations. Defaults to 500.

    Returns:
        candidate configurations
    """
    N, D = X.shape
    coreset = [i for i in range(N)]
    container = []
    best_candidates = []
    r=0
    last_coreset_size = N
    Tfact = 0.95 #default zero

    counter = 0  # Counter to track unchanged value occurrences

    while(len(coreset) > Bmax):
             
        container = container[:L]
        r=r+1

        start = time.perf_counter()
        print(f'this is {r}th shuffle')

        random.shuffle(coreset)
        shuflled_blocks = divide_into_blocks(coreset, M)
        i = 0
        print(f'the number of block in this process is {len(shuflled_blocks)}')
        for block in shuflled_blocks:
            i = i+1
            start = time.perf_counter()
            res = Deep_ICE(block, X, t, K, 500, P=False)
            end = time.perf_counter()
            print(f'the total time of {i}th block is {end-start}')
            res_block = (res[0],res[1],block)
            
            # add candidate configuration to the heap
            best_candidates = add_cnfg(best_candidates, res_block, num_candidates)
            container = add_cnfg(container, res_block, L)
            if container[0][1] == 0:
                return container[0]
        print(f'the best configuration in this shuffle{container[0]}')
        end = time.perf_counter()
        print(f'the total time for {r}th shuffle is {end-start}')
        coreset = []
        coreset = [(coreset + i[2]) for i in container]
        coreset = unique(coreset)
        print(f'the coresetsize in last shuffle is {last_coreset_size}')
        print(f'the coresetsize in this shuffle has size {len(coreset)}')


        if last_coreset_size - len(coreset) < threshold :
            counter += 1
        else:
            counter = 0  # Reset counter if value changes
        print(f'the maximal unchange is {max_unchanged}')
        print(f'the counter  in this shuffle is {counter}')

        # adjusting the parameters during the algorithm running time
        if counter >= max_unchanged:
            L = int(L*Tfact)
            if len(coreset)<=100:
                # M=39
                Tfact = 0.95
                threshold = 1
                # M = 6
                # max_unchanged = 200
                max_unchanged = 20
            if len(coreset)<=150:
                # M=39
                Tfact = 0.95
                threshold = 1
                # M = 5
                # max_unchanged = 150
                max_unchanged = 15
            elif len(coreset)<=300:
                # M=39
                Tfact = 0.95
                threshold = 2
                # M = 29
                # max_unchanged = 100
                max_unchanged = 10
            elif len(coreset)<=500:
                # M=39
                Tfact = 0.95
                threshold = 3
                # max_unchanged = 50
                max_unchanged = 5

 
            counter = 0  # Reset counter after action

        print(f'the new container has size {L}')
        last_coreset_size = len(coreset)
        print('\n')

    return best_candidates


