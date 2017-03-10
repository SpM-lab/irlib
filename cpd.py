import numpy as np
import scipy
#import sktensor
from tensorly.decomposition import parafac
from tensorly.kruskal import kruskal_to_tensor
import hosvd

nl = 30
S = np.loadtxt('S.txt')
S = S.reshape((nl,nl,nl))

u,s,d = hosvd.HOSVD(S)

#print(s.shape)
for l in range(nl):
    print l, np.abs(s[l,l,l])

for rank in [20, 40, 60, 80, 100, 200, 500, 1000]:
    factors = parafac(S, rank=rank)
    #print(factors[0].shape)
    
    #S_r = np.zeros_like(S)
    #norm_list = []
    #for ir in range(rank):
        #S_r += np.einsum('i,j,k->ijk', factors[0][:,ir], factors[1][:,ir], factors[2][:,ir])
        #print ir, np.linalg.norm(S_r-S)
        #norm_list.append(np.linalg.norm(factors[0][:,ir])*np.linalg.norm(factors[1][:,ir])*np.linalg.norm(factors[2][:,ir]))
        #print ir, np.linalg.norm(factors[0][:,ir])*np.linalg.norm(factors[1][:,ir])*np.linalg.norm(factors[2][:,ir])
    #norm_list = np.array(norm_list)
    #print norm_list
    #norm_list = np.sort(norm_list)
    #print norm_list
    #for ir in range(rank):
        #print ir, norm_list[ir]

    #S_r = np.einsum('il,jl,kl->ijk', factors[0], factors[1], factors[2])
    S_r = kruskal_to_tensor(factors)
    print rank, np.linalg.norm(S_r - S)/np.linalg.norm(S)
    #print(S[0:2,0:2,0:2])
    #print(S_r[0:2,0:2,0:2])
    
    #p, fit, itr, exectimes = sktensor.cp.als(S, 3)
    #print(p)
