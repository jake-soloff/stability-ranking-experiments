import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import time
from tqdm import tqdm

# functions for inflated argmax
def proj(w, j, eps):
    d = len(w)
    wj = w[j]
    w1 = np.delete(w,j)
    means = (w[j] + np.hstack([0,np.cumsum(w1)]))/np.arange(1,d+1)
    shrink_val = means - (eps/np.sqrt(2))/np.arange(1,d+1)
    ind = np.min(np.where(np.hstack([w1, -float('Inf')]) < shrink_val))
    w_proj = w.copy()
    if ind>0:
        w_proj[w_proj > shrink_val[ind]] = shrink_val[ind]
        w_proj[j] = shrink_val[ind] + eps/np.sqrt(2)
    return(w_proj, np.linalg.norm(w-w_proj))

def sargmax(eps, w):
    d = len(w)
    k = 0
    
    for i in range(d):
        _, dist = proj(w, i, eps)
        if dist < eps:
            k+=1
        else:
            break
    return k

# experiment
def run_exp_fillin(filter_threshold, eps, Klist, fulldf, n, N_rep):
    popular_movies = Full_rating_count[Full_rating_count >= filter_threshold].index
    df = Fulldf[Fulldf['movie_id'].isin(popular_movies)]
    unique_user_ids = df['user_id'].unique()
    
    Stab_delta = np.zeros((N_rep,len(Klist),2))
    Jaccads = np.zeros((N_rep,len(Klist),2))
    Size = np.zeros((N_rep, len(Klist)))
    Score_stab = np.zeros((N_rep,3))
    
    rng = np.random.default_rng(seed=124)
    
    for t in tqdm(range(N_rep)):
        # sample a dataset from the population
        sampled_user_ids = rng.choice(unique_user_ids, size=n, replace=False)
        data = df[df['user_id'].isin(sampled_user_ids)]
        global_count = data.groupby('movie_id')['rating'].count()
        # if there's any movie having zero rating, resample
        while global_count.min()<1:
            sampled_user_ids = rng.choice(unique_user_ids, size=n, replace=False)
            data = df[df['user_id'].isin(sampled_user_ids)]
            global_count = data.groupby('movie_id')['rating'].count()
        
        score_stab = np.zeros(n)
        stab_deltas = np.zeros((n,len(Klist),2))
        Jac = np.zeros((n,len(Klist), 2))
        
        # all users stats
        global_sum = data.groupby('movie_id')['rating'].sum()
        
        all_mean_pd = data.groupby('movie_id')['rating'].mean()
        movies = all_mean_pd.index
        all_mean = all_mean_pd.to_numpy() #all-user average rating (unsorted)
        sort_ind_all = np.argsort(-all_mean) # in descending order
        sort_mean_all = all_mean[sort_ind_all]
        
        # loop for user leave-one-out (LOO)
        for i in range(n):
            user_id = sampled_user_ids[i]
            user_data = data[data['user_id'] == user_id]
            user_sum = user_data.groupby('movie_id')['rating'].sum().reindex(movies, fill_value=0)
            user_count = user_data.groupby('movie_id')['rating'].count().reindex(movies, fill_value=0)
            
            leave_out_sum = global_sum.to_numpy() - user_sum.to_numpy()
            leave_out_count = global_count.to_numpy() - user_count.to_numpy()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                loo_mean = leave_out_sum / leave_out_count
            loo_mean[leave_out_count == 0] = 3 # NA filled with 3

            score_stab[i] = np.linalg.norm(all_mean - loo_mean)
            
            sort_ind_loo = np.argsort(-loo_mean)
            sort_mean_loo = loo_mean[sort_ind_loo]
            
            # for current w and w\-i
            for k_ind in range(len(Klist)):
                k = Klist[k_ind]
                
                topk = set(sort_ind_all[l] for l in range(k))
                
                max_ind = sargmax(eps, sort_mean_all[(k-1):])
                stopk = set(sort_ind_all[l] for l in range(k-1, k+max_ind-1)) | topk  
                Size[t, k_ind] = len(stopk)
                
                # evaluate topk
                loo_topk = set(sort_ind_loo[l] for l in range(k))
                cap = len(topk & loo_topk)
                cup = len(topk | loo_topk)
                Jac[i,k_ind,1] = cap/cup
                if cap<k:
                    stab_deltas[i,k_ind,1] = 1
                
                # evaluate inflated topk
                max_ind_loo = sargmax(eps, sort_mean_loo[(k-1):])
                loo_stopk = set(sort_ind_loo[l] for l in range(k-1, k+max_ind_loo-1)) | loo_topk
                cap = len(stopk & loo_stopk)
                cup = len(stopk | loo_stopk)
                Jac[i,k_ind,0] = cap/cup
                if cap<k:
                    stab_deltas[i,k_ind,0] = 1
        
        # average over all LOO
        Score_stab[t,0] = np.mean(score_stab); Score_stab[t,1] = np.max(score_stab)
        Score_stab[t,2] = (score_stab > eps).mean()
        
        Stab_delta[t,:,:] = np.mean(stab_deltas, axis=0)
        Jaccads[t,:,:] = np.mean(Jac, axis = 0)
        
    
    # save the result (Save multiple arrays into one file)
    filename_all = f"topk_results_eps_{eps}_threshold_{filter_threshold}_n_{n}_Nrep_{N_rep}_fillin3.npz"
    np.savez(filename_all, array1=Stab_delta, array2=Jaccads, array3=Size, array4=Score_stab)
    # final result
    mean_Stab_delta = np.mean(Stab_delta, axis=0)  # shape (K, 2)
    mean_Jaccads = np.mean(Jaccads, axis=0)
    mean_size = np.mean(Size, axis=0)
    se_Stab_delta = np.std(Stab_delta, axis=0)/ np.sqrt(N_rep)
    se_Jaccads = np.std(Jaccads, axis=0)/ np.sqrt(N_rep)
    se_size = np.std(Size, axis=0)/ np.sqrt(N_rep)
    
    max_delta_stopk = np.max(Stab_delta[:,:,0], axis=0) # shape (K,)
    max_delta_topk = np.max(Stab_delta[:,:,1], axis=0) # shape (K,)

    summary = np.stack([np.array(Klist), mean_Stab_delta[:,0], mean_Jaccads[:,0], mean_size,
                   se_Stab_delta[:,0], se_Jaccads[:,0], se_size,
                   mean_Stab_delta[:,1], mean_Jaccads[:,1], se_Stab_delta[:,1], se_Jaccads[:,1],
                   max_delta_stopk, max_delta_topk], axis=1)
    df_summary = pd.DataFrame(summary, columns=['K', 'mean_Stab_delta_stopk', 'mean_Jaccads_stopk', 'mean_size',
                                           'SE_Stab_delta_stopk', 'SE_Jaccads_stopk', 'SE_size',
                                           'mean_Stab_delta_topk', 'mean_Jaccads_topk', 
                                            'SE_Stab_delta_topk', 'SE_Jaccads_topk',
                                           'max_delta_stopk', 'max_delta_topk'])
    filename_sum = f"topk_summary_eps_{eps}_threshold_{filter_threshold}_n_{n}_Nrep_{N_rep}_fillin3.csv"
    df_summary.to_csv(filename_sum, index=False)
    return Stab_delta, df_summary


# Load the data
Fulldf = pd.read_csv('fullData.csv')
Full_rating_count = Fulldf.groupby('movie_id')['rating'].count()


# Experiment settings 
filter_threshold = 200
N_rep = 100; 
Klist = [3,5,10,20,50]
n = 10000
eps = 0.0001

# run the experiment
delta_j, result_summary = run_exp_fillin(filter_threshold, eps, Klist, Fulldf, n, N_rep)


# plot the results for n=10^4, eps=10^{-4} and threshold = 200
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## specify plot options 
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams.update({
    'axes.linewidth' : 2,
    'font.size': 22,
    "text.usetex": True,
    'font.family': 'serif', 
    'font.serif': ['Computer Modern'],
    'text.latex.preamble' : r'\usepackage{amsmath,amsfonts}'})

## custom color palette
lblue = (40/255,103/255,178/255)
cred  = (177/255, 4/255, 14/255)

# plot survival functions
def plot_sf(arr, c, lab=None, ls='-'):
    size = len(arr)
    plt.step(np.sort(arr)[::-1], np.arange(size)/size, 
             where='post', c=c, 
             label=lab, ls=ls)


# Load the .npz file
fullResults = np.load('topk_results_eps_0.0001_threshold_200_n_10000_Nrep_100_fillin3.npz')
# Retrieve the array stored under 'array1', which is Stab_delta
delta_j = fullResults['array1']


Klist = [3,5,10,20,50]

for k_ind in range(len(Klist)):
    k = Klist[k_ind]
    
    plt.figure(figsize=(10,5))
    plot_sf(delta_j[:,k_ind,1], cred, 
        'top-$k$')
    plot_sf(delta_j[:,k_ind,0], lblue, 
        '$\\textnormal{top-$k$}^{(\\varepsilon)}$')
    
    plt.semilogy()
    plt.legend(loc='upper right')
    
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    
    plt.xticks([0, .4, 0.8])
    
    plt.xlabel('$\\delta$', fontsize=20)
    plt.ylabel('$\\frac{1}{N}\sum_{j\in [N]}1\\{\\delta_j > \\delta\\}$', 
           fontsize=20, labelpad=20)
    
    plt.ylim([0,2])
    plt.xlim([0,0.8])
    
    plt.title(fr"tability comparison with $k={k}$",fontsize=20)
    plt.tight_layout()
    
    plotname = f"topk_instability_k{k}.pdf"
    plt.savefig(plotname)
    plt.show()

