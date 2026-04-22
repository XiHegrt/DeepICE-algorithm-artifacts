import torch
import math


def complement(x, y):
    return [item for item in y if item not in x]


def asgns_gen(K):
    asgns = [[]]
    for k in range(K):
        pos_asgns = [asgn + [1] for asgn in asgns]
        neg_asgns = [asgn + [-1] for asgn in asgns]
        asgns = pos_asgns + neg_asgns
    return asgns

def gen_N_choose_K_matrix(N,K):

    global nchoosek 
    nchoosek = torch.zeros((N+1,K+1), dtype=torch.int32)
    for n in range(N+1):
        for k in range(K+1):
            nchoosek[n,k] = math.comb(n,k)
    return nchoosek

def unranking(r,N,K,inds):
    r = int(r)
    if K == 0:
       return []
    else:
        x = N
        while math.comb(x,K) > r:
            x = x-1
        return unranking(math.comb(x+1,K)-r-1,N,K-1,inds)+[inds[x]]


def right_upd(a, x:list):
    if x == []:
        return [[a]]
    else:
        return [b + [a] for b in reversed(x)]


def kcombs_int(K, N:int):
    cnfgs = [[[]]]+[[] for _ in range(K)]
    for n in range(N):
        temp = [cnfg.clone() for cnfg in cnfgs]
        for k in range(min(K,n+1)):
            cnfgs[k+1] = cnfgs[k+1]+(right_upd(n, temp[k]))

    return cnfgs


def right_upd_array(a, M, cs:torch.tensor):
    """update method for CGC combination generation
    Args:
        a (int): update value
        M (int): the number of configurations in x, equivalent to n+1 choose k
        cs (torch.tensor): combinations of size k
    Returns:
        torch.tensor: updated combinations
    """
    if cs.size(1)==0:
        cs_new = torch.tensor([[a]])
    else:
        a_M = torch.full((M, 1), a)
        cs_new = cs.flip(0)
        cs_new = torch.cat((cs_new, a_M), dim=1)
    return cs_new

def upd_array(a, M, cs:torch.tensor):
    """update method for ordinary combination generation
    Args:
        a (int): update value
        M (int): the number of configurations in x, equivalent to n+1 choose k
        cs (torch.tensor): combinations of size k
    Returns:
        torch.tensor: updated combinations
    """
    if cs.size(1)==0:
        cs_new = torch.tensor([[a]])
    else:
        a_M = torch.full((M, 1), a)
        cs_new = torch.cat((cs, a_M), dim=1)
    return cs_new


def kcombs_int_array(K, N):
    cnfgs = [torch.tensor([[]]) for _ in range(K+1)]
    for n in range(N):
        temp = [cnfg.clone() for cnfg in cnfgs]
        for k in range(min(K,n+1)):
            M=math.comb(n, k)
            if temp[k+1].size(1)==0:
                cnfgs[k+1] = right_upd_array(n, M, temp[k])
            else:
                cnfgs[k+1] = torch.cat((cnfgs[k+1], right_upd_array(n, M, temp[k])),dim=0)
    return cnfgs

def kcombs_iter_array(K,N, X:list):
    cnfgs = [torch.tensor([[]]) for _ in range(K+1)]
    for n in range(N):
        temp = [cnfg.clone() for cnfg in cnfgs]
        for k in range(min(K,n+1)):
            M=math.comb(n, k)
            if temp[k+1].size(1)==0:
                cnfgs[k+1] = upd_array(X[n], M, temp[k])
            else:
                cnfgs[k+1] = torch.cat((cnfgs[k+1], upd_array(X[n], M, temp[k])),dim=0)
    return cnfgs


def cross_join_array(cs1:torch.tensor, cs2:torch.tensor):
    if cs1.size(1)==0 or cs2.size(1)==0:
        return torch.tensor([[]])

    R1, C1 = cs1.shape
    R2, C2 = cs2.shape
    if R1 < R2:
        R_min = R1
    else:
        R_min = R2

    R_new = R1 * R2
    C_new = C1 + C2
    cs_new = torch.zeros((R_new, C_new), dtype=int) # integer from -32,768 to +32,767)
    for i in range(R_min):
        if R_min == R1:
           temp = cs1[i,:].unsqueeze(0).expand(R2, -1) #(R2, C1)
           cs_new[i*R2:(i+1)*R2,:] = torch.cat((temp, cs2),dim=1) #cs2:(R2,C2), cs_new(R2,C1+C2)
        else:
            temp = cs2[i,:].unsqueeze(0).expand(R1, -1) #(1000,3)
            cs_new[i*R1:(i+1)*R1,:] = torch.cat((cs1,temp),dim=1)
    return cs_new

def convol_filt_array(f, k, x:list, y:list):
    """ the naive implementation of convolution with filtering,
        can be accelarated by applying FFT

    Args:
        f : a binary function
        k (int) : the filter condition, filter out all (i,j) with i+j>=k
        a (list) : input list
        b (list) : input list

    Returns:
        list of lists: the covolution of a and b after filtering
    """
    # Length of input sequences
    len_x, len_y = len(x), len(y)

    # Initialize result list with zeros
    result = [torch.tensor([[]]) for _ in range(k+1)]
    
    # Perform element-wise convolution
    for i in range(len_x):
        for j in range(len_y):
            if (i+j) > k:
               continue
            if i == 0:
                if y[j].size(1) == 0:
                    continue
                elif result[i + j].size(1)==0:
                   result[i + j] = y[j]
                else:
                    result[i + j] = torch.cat((result[i + j], y[j]),dim=0)
            elif j == 0:
                if x[i].size(1) == 0:
                    continue
                elif result[i + j].size(1)==0:
                   result[i + j] = x[i]
                else:
                    result[i + j] = torch.cat((result[i + j], x[i]),dim=0)
            else:
                cs_new = cross_join_array(x[i],y[j])
                if cs_new.size(1) == 0:
                    continue
                elif result[i + j].size(1)==0:
                   result[i + j] = cs_new
                else:
                    result[i + j] = torch.cat((result[i + j], cs_new),dim=0)
    return result

################## auxiliary functions for coreset ##################


def divide_into_blocks(my_list, block_size):
    return [my_list[i:i + block_size] for i in range(0, len(my_list), block_size)]


def flatten(xs):
   return [a for x in xs for a in x]

def unique(xs):
    x = flatten(xs)
    return list(set(x))

