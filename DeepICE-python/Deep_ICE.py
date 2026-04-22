from math import inf,comb
from auxfuncs_array import *
import torch

# separete the hyperplane generation process


def gen_models_GPU(hyper_asgns, cs,inds, n1, X_cuda):
    """Dynamically generate hyperplanes for the given combinations of data points

    Args:
        hyper_asgns (_type_): _description_
        cs (_type_): _description_
        inds (_type_): _description_
        n1 (_type_): _description_
        X_cuda (_type_): _description_

    """
    N, D = X_cuda.shape
    cs_cuda = cs.cuda()
    M = cs.size(0)

    if D <= block_size/2: 
        A_chunk = X_cuda[cs_cuda]
        
        # Reshape the extracted submatrices into the desired tensor shape 
        b_chunk = torch.ones(M, D, 1, device='cuda',dtype=torch.float64) 
    
        # Solve the batched linear systems
        ws = torch.linalg.solve(A_chunk, b_chunk) 
        ws = ws.squeeze(-1) #(M,D)
        ones = torch.ones(M, 1).cuda() 

        hyper_dists =  torch.matmul(ws, X_cuda.T) - ones 

        hyper_asgns_temp =  torch.sign(hyper_dists)

        # Use advanced indexing to set the elements at the specified indices to 0
        hyper_asgns_temp[torch.arange(M).unsqueeze(1), cs_cuda] = 0 


        hyper_asgns[n1:(n1+M),:] = hyper_asgns_temp

    else:

        inds_cuda = torch.tensor(inds).cuda()
        inds_mat = inds_cuda.repeat(M,1) 
        cs_cuda = cs.cuda()

        # Create a mask for each element in inds_mat to check if it is in cs_cuda
        matches = (cs_cuda.unsqueeze(2) == inds_cuda) 
        matches = torch.sum(matches, dim=1, dtype=torch.bool) 
        cs_cuda_complement = inds_mat[~matches].view(M,D) 


        A_chunk = X_cuda[cs_cuda_complement] 
        
        # Reshape the extracted submatrices into the desired tensor shape 
        b_chunk = torch.ones(M, D, 1, device='cuda',dtype=torch.float64) 
    
        # Solve the batched linear systems
        ws = torch.linalg.solve(A_chunk, b_chunk)
        ws = ws.squeeze(-1)
        ones = torch.ones(M, 1).cuda() 
        
        hyper_dists =  torch.matmul(ws, X_cuda.T)-ones

        hyper_asgns_temp =  torch.sign(hyper_dists)

        # Use advanced indexing to set the elements at the specified indices to 0
        hyper_asgns_temp[torch.arange(M).unsqueeze(1), cs_cuda_complement] = 0 

        hyper_asgns[n1:(n1+M),:] = hyper_asgns_temp
    return hyper_asgns


def evalfilt_GPU(hyper_asgns, ncs, asgns, t_cuda, K, ncs_block):
    """Evaluate the configurations in ncs and select the best one

    Args:
        hyper_asgns: Pre-computed hyperplane assignments
        ncs (M,K): Candidate nested combinations
        asgns (2^K,K): All possible orientation of hyperplanes
        t_cuda (N,): Label vector, consists of -1 and 1
        K (int): Number of hyperplanes
        ncs_block (int): The block size for parallel computation

    Returns:
        the best nested combination and its loss
    """

    ncs_cuda = ncs.cuda()
    len_ncs = ncs_cuda.size(0)
    N = t_cuda.size(0)

    num_streams = math.ceil(len_ncs / ncs_block)

    ncs_chunks = [ncs[i:i + ncs_block] for i in range(0, len_ncs, ncs_block)]

    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    cnfgs = []

    for chunk, stream in zip(ncs_chunks, streams):
        with torch.cuda.stream(stream):

            M = chunk.size(0)       

            ksvs = hyper_asgns[chunk] 

            best_losses = torch.full((M,), N, dtype=torch.int32).cuda()
            for i in range(2**(K-1)):
                asgn = asgns[i]
                asgn_tensor = torch.tensor(asgn).cuda() 

                # multiply asgn_tensor with each (K,M) matrix in ksvs along the row 
                asgn_tensor = asgn_tensor.view(1, K, 1)
                ksvs_asgn = ksvs*asgn_tensor

                #  maxout function, all sign vector greater than 0 predict to positive, else are negative class
                svs, min_index = torch.max(ksvs_asgn, dim=1)  # (M, 1, N)

                # calculating positive losses
                sum_labelss = svs + t_cuda
                losses_pos = torch.sum(sum_labelss == 0, dim=1, dtype=torch.int32) #size (M,)
                temp_pos = torch.stack((losses_pos, best_losses), dim=1)
                best_losses, min_index = torch.min(temp_pos,dim=1)
                
                # calculate negative losses
                ksvs_asgn_neg = -ksvs_asgn
                svs_neg, min_index = torch.max(ksvs_asgn_neg, dim=1)  # (M, 1, N)
                sum_labels_neg = svs_neg + t_cuda
                losses_neg = torch.sum(sum_labels_neg == 0, dim=1, dtype=torch.int32)
                temp_neg = torch.stack((best_losses, losses_neg), dim=1)


                best_losses, min_index = torch.min(temp_neg,dim=1)

                candidate_loss, candidate_ind =  torch.min(best_losses,dim=0)
                   
            cnfg = (chunk[candidate_ind], candidate_loss)
            cnfgs.append(cnfg)
    cnfgs = min(cnfgs, key=lambda x: x[1])
    return cnfgs


def Deep_ICE(inds, X, t, K,ncs_block, P=True):
    """ ERM algorithm for rank-K maxout neural network

    Args:
        inds (list): Data indexes. If run to completion, then inds = [i for i in range(N)]
        X (N,D): Input data, shape (N,D)
        t (N,): Label vector, consists of -1 and 1
        K (int): Number of hyperplanes
        ncs_block (int): The block size for parallel computation
        P (bool): Print the process, default is True

    Returns:
        ERM solution, a tuple of (configuration, loss)
    """
    asgns = asgns_gen(K)

    global block_size
    block_size = len(inds)

    t = torch.tensor(t)
    X = torch.tensor(X).to(torch.float64)
    # X = X_hom[:, :-1]
    N,D = X.shape
    N_block = len(inds)

    if block_size <= D:
        return (None,inf)

    # move to cuda
    X_cuda = X.cuda()
    t_cuda = t.cuda()


    opt_cnfg = (None,inf)
    best_cnfg = (None,inf)
    n = -1
    if D <= block_size/2: # check if dimension greater than N/2

        hyper_asgns = torch.zeros((math.comb(block_size,D), N), dtype=torch.int8).cuda()
        # initialization
        css = [torch.tensor([[]]) for _ in range(D+1)] # list used to store combinations of data items, type [[[int]]]
        ncss = [torch.tensor([[]]) for _ in range(K+1)]


        for i in inds:
            n +=1
            if P == True:
               print(f'This is stage {n}')
            temp = [cs.clone() for cs in css]
            for d in range(min(D,n+1)):
                M= comb(n, d)
                if temp[d+1].size(1)==0:
                    css[d+1] = right_upd_array(i, M, temp[d])
                else:
                    css[d+1] = torch.cat((css[d+1], right_upd_array(i, M, temp[d])),dim=0)
     
            if css[D].size(1)!=0: # if D-comb of data is not empty we can generate hyperplanes
                n1 = comb(n,D)
                n2 = comb(n,D) + comb(n,D-1)
                # print(n1)
                # generate hyperplanes and store them in N^D candidate list
                hyper_asgns = gen_models_GPU(hyper_asgns, css[D], inds, n1, X_cuda) 
                css[D] = torch.tensor([[]])
                # print(hyper_asgns)

            
                # genererated nested combinations, where D-combinations are represented by its rank
                # ,corresponding to each rows in matrix hyper_asgns
                ints = list(range(n1, n2))
                ncss_new = kcombs_iter_array(K, n2-n1, ints)
                ncss = convol_filt_array(cross_join_array, K, ncss, ncss_new )

                if P == True:
                    print(f'the number of ncss in this stage is {ncss[K].size(0)}')
    
                if ncss[K].size(1)!=0:
                    # select the best configuration
                    best_cnfg = evalfilt_GPU(hyper_asgns, ncss[K], asgns, t_cuda, K, ncs_block)
                    # clear out evaluated configurations
                    ncss[K] = torch.tensor([[]])
    
                # check if the best solution generated in this step is better than the current optimal
                if best_cnfg[1] < opt_cnfg[1]:
                    opt_cnfg = best_cnfg
                    opt_cnfg = ([unranking(i, N, D, inds) for i in opt_cnfg[0]],opt_cnfg[1])
                    if P == True:
                        print(f'The optimal configuration is updated as {opt_cnfg}')
                # print(f'The total number of configurations has been explored {math.comb(n2, K)}')
    else:
        D_complement = block_size-D
        # initialization
        css = [torch.tensor([[]]) for _ in range(D_complement+1)] # list used to store combinations of data items, type [[[int]]]
        ncss = [torch.tensor([[]]) for _ in range(K+1)]

        hyper_asgns = torch.zeros((math.comb(block_size,D_complement), N), dtype=torch.int8).cuda()
        for i in inds:
            n +=1
            if P == True:
                print(f'This is stage {n}')
            temp = [cs.clone() for cs in css]
            for d in range(min(D_complement,n+1)):
                M=math.comb(n, d)
                if temp[d+1].size(1)==0:
                    css[d+1] = right_upd_array(i, M, temp[d])
                else:
                    css[d+1] = torch.cat((css[d+1], right_upd_array(i, M, temp[d])),dim=0)

            if css[D_complement].size(1)!=0: # if D-comb of data is not empty we can generate hyperplanes
                n1 = comb(n,D_complement)
                n2 = comb(n,D_complement) + comb(n,D_complement-1)

    
                # generate hyperplanes and store them in N^D candidate list
                hyper_asgns = gen_models_GPU(hyper_asgns, css[D_complement], inds, n1, X_cuda) 
                css[D_complement] = torch.tensor([[]])



                # genererated nested combinations, where D-combinations are represented by its rank
                # ,corresponding to each rows in matrix hyper_asgns
                ints = list(range(n1, n2))
                ncss_new = kcombs_iter_array(K, n2-n1, ints)
                ncss = convol_filt_array(cross_join_array, K, ncss, ncss_new )

                if P == True:
                    print(f'the number of ncss in this stage is {ncss[K].size(0)}')
                # print(ncss[K-1].size(0))
    
                if ncss[K].size(1)!=0:
                    # select the best configuration
                    best_cnfg = evalfilt_GPU(hyper_asgns, ncss[K],asgns, t_cuda, K, ncs_block)
                    # clear out evaluated configurations
                    ncss[K] = torch.tensor([[]])
    
                # check if the best solution generated in this step is better than the current optimal
                if best_cnfg[1] < opt_cnfg[1]:
                    opt_cnfg = best_cnfg
                    opt_cnfg = ([unranking(i, N, D_complement, inds) for i in opt_cnfg[0]],opt_cnfg[1])

                    # calculating complements
                    opt_cnfg = ([complement(i,inds) for i in opt_cnfg[0]],opt_cnfg[1])


                    if P == True:
                        print(f'The optimal configuration is updated as {opt_cnfg}')       
    return opt_cnfg

