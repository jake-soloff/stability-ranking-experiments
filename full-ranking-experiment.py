import numpy as np
np.random.seed(1234567891)

from scipy.linalg import sqrtm

def proj(w, j, eps):
    d = len(w)
    wj = w[j]
    w1 = np.sort(np.delete(w,j))[::-1]
    means = (w[j] + np.hstack([0,np.cumsum(w1)]))/np.arange(1,d+1)
    shrink_val = means - (eps/np.sqrt(2))/np.arange(1,d+1)
    ind = np.min(np.where(np.hstack([w1, -float('Inf')]) < shrink_val))
    w_proj = w.copy()
    if ind>0:
        w_proj[w_proj > shrink_val[ind]] = shrink_val[ind]
        w_proj[j] = shrink_val[ind] + eps/np.sqrt(2)
    return(w_proj, np.linalg.norm(w-w_proj))

def eps_argmax(eps, w):
    d = len(w)
    k = 0
    o = np.argsort(-w)
    for i in range(d):
        j = o[i]
        _, dist = proj(w, j, eps)
        if dist < eps:
            k+=1
        else:
            break
    return o[:k]

def eps_ranking(eps, w):
    L = len(w)
    results = set()
    
    def backtrack(current_perm, remaining_w, remaining_indices):
        if len(current_perm) == L:
            results.add(tuple(current_perm))
            return

        eligible_local = eps_argmax(eps, remaining_w) # eligible player id
        for local_i in eligible_local:
            global_i = remaining_indices[local_i]
            
            # New w and index list without the selected index
            new_w = np.delete(remaining_w, local_i)
            new_indices = remaining_indices[:local_i] + remaining_indices[local_i+1:]

            backtrack(current_perm + [global_i], new_w, new_indices)

    backtrack([], w.copy(), list(range(L)))
    return results

# Return ridge solution with ||bhat|| ≈ R
def cridge(X, y, R, tol=1e-6):
    n, p = X.shape
    
    if n > p:
        # Return OLS if it satisfies the radius constraint
        ols = np.linalg.inv(X.T@X)@X.T@y
        if np.linalg.norm(ols) <= R:
            return(ols)
        else: 
            # Initialize lower bound of lambda
            l = 0
            bhat_l = ols
    else:
        # Initialize lower bound of lambda
        l = .5
        bhat_l = np.linalg.inv(X.T@X+l*np.eye(p))@X.T@y

    # Find lower bound: decrease regularization until norm > R
    while np.linalg.norm(bhat_l) <= R:
        l /= 2
        bhat_l = np.linalg.inv(X.T@X+l*np.eye(p))@X.T@y
    
    # Find upper bound: increase regularization until norm < R
    r = 1
    bhat_r = np.linalg.inv(X.T@X+r*np.eye(p))@X.T@y    
    while np.linalg.norm(bhat_r) > R:
        r *= 2
        bhat_r = np.linalg.inv(X.T@X+r*np.eye(p))@X.T@y

    # Binary search to find lambda such that ||bhat|| ≈ R
    while ((np.linalg.norm(bhat_l) - R) > tol) or ((np.linalg.norm(bhat_r) - R) < -tol):
        m = (l+r)/2
        bhat = np.linalg.inv(X.T@X+m*np.eye(p))@X.T@y
        if np.linalg.norm(bhat) > 1:
            bhat_l = bhat
            l = m
        else:
            bhat_r = bhat
            r = m
    return(bhat)
    
    
eps = .05 # Inflation parameter for eps_ranking

# Lists to store results across iterations
sizes = []         # Sizes of inflated selection sets
deltas = []        # Instability of standard ranking
deltas_infl = []   # Instability of eps-inflated ranking

niter = 1000 # Number of simulation iterations
for _ in range(niter):
    n, p = 50, 5  # Sample size and dimension

    # Generate AR(1) covariance structure
    rho = .5
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    rtcov = sqrtm(cov)
    
    # Generate data matrix X ~ N(0, cov)
    X = np.random.randn(n, p) @ rtcov
    
    # True signal vector beta
    beta = np.arange(p)
    beta = beta/np.linalg.norm(beta)

    # Generate response y = Xβ + noise
    y = X@beta + np.random.randn(n)

    # Compute constrained ridge estimate
    w = np.abs(cridge(X, y, 1))
    
    # Compute standard and eps-inflated rankings
    perm = np.argsort(w)
    perm_eps = eps_ranking(eps, w)

    # Initialize counters
    ct = 0         # Count of LOO rank instability
    ct_eps = 0     # Count of LOO instability under eps-inflated ranking
    size_eps = len(perm_eps)  #    Size of eps-inflated selection set

    inds = np.arange(n)
    for i in range(n):
        ic = np.delete(inds, i) # LOO index set
        
        # Recompute estimate with one sample left out
        w_i = np.abs(cridge(X[ic], y[ic], 1))
        
        # Compare exact rankings
        perm_i = np.argsort(w_i)
        ct += ~np.all(perm==perm_i)

        # Compare eps-inflated rankings
        perm_eps_i = eps_ranking(eps, w_i)
        ct_eps += (len(perm_eps & perm_eps_i) == 0)
        
    # Store instability rates and selection set size
    deltas.append(ct/n)
    deltas_infl.append(ct_eps/n)
    sizes.append(size_eps)
    
if __name__ == "__main__":
    # Report summary statistics
    print('max delta_j:', np.max(deltas), 
          '\nmax delta_j (eps-infl.):',  np.max(deltas_infl), 
          '\navg delta_j:', np.mean(deltas), '(%.4f)' % (np.std(deltas)/np.sqrt(niter)),
          '\navg delta_j (eps-infl.):',  np.mean(deltas_infl), '(%.4f)' % (np.std(deltas_infl)/np.sqrt(niter)),
          '\nsize (eps-inflated):', np.mean(sizes), '(%.4f)' % (np.std(sizes)/np.sqrt(niter)),)
