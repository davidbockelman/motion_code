import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize

from sparse_gp import *
from utils import *

def optimize_motion_codes(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1, R=3):
    '''
    Main algorithm to optimize all variables for the Motion Code model with mixture of experts.
    
    Parameters
    ----------
    R: int
        Number of global motion codes (experts)
    '''
    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, R, m, latent_dim, Q)

    # Initialize parameters
    X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), latent_dim, axis=0).swapaxes(0, 1)
    # Initialize expert codes with random values so they start different
    Z_start = np.random.randn(R, latent_dim) * 0.1  # R global motion codes with small random initialization
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))
    # Initialize expert weights logits (before softmax) - start with uniform weights
    expert_weights_logits_start = np.zeros((num_motion, R))  # Uniform after softmax

    # Optimize X_m, Z, kernel parameters (Sigma, W), and expert weights
    res = minimize(fun=elbo_fn(X_list, Y_list, labels, sigma_y, dims),
        x0 = pack_params([X_m_start, Z_start, Sigma_start, W_start, expert_weights_logits_start]),
        method='L-BFGS-B', jac=True)
    X_m, Z, Sigma, W, expert_weights_logits = unpack_params(res.x, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)
    # Apply softmax to get actual weights (per class, so each row sums to 1)
    expert_weights = np.array([softmax(expert_weights_logits[k]) for k in range(num_motion)])

    # We now optimize distribution params for each motion-expert combination
    # Structure: mu_ms[k][r], A_ms[k][r], K_mm_invs[k][r] for process k and expert r
    mu_ms = []; A_ms = []; K_mm_invs = []

    # All timeseries of the same motion is put into a list, an element of X_motion_lists and Y_motion_lists
    X_motion_lists = []; Y_motion_lists = []
    for _ in range(num_motion):
        X_motion_lists.append([]); Y_motion_lists.append([])
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    # For each motion k and each expert r, compute optimal distribution params
    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_ms_k = []; A_ms_k = []; K_mm_invs_k = []
        for r in range(R):
            mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m@Z[r]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params)
            mu_ms_k.append(mu_m); A_ms_k.append(A_m); K_mm_invs_k.append(K_mm_inv)
        mu_ms.append(mu_ms_k); A_ms.append(A_ms_k); K_mm_invs.append(K_mm_invs_k)
    
    # Save model to path.
    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W, 
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs, 
             'expert_weights': expert_weights, 'R': R}
    np.save(model_path, model)
    return

def classify_predict_helper(X_test, Y_test, kernel_params_all_motions, X_m, Z, mu_ms, A_ms, K_mm_invs, mode='dt', R=None, expert_weights=None):
    """
    Classify by calculate distance between inducing (mean) values and interpolated test values at inducing pts.
    Uses mixture of experts with R global motion codes and learned static weights per class.
    """
    num_motion = len(kernel_params_all_motions)
    # Infer R from structure if not provided
    if R is None:
        if isinstance(mu_ms[0], list):
            R = len(mu_ms[0])
        else:
            R = 1  # Fallback to single expert (backward compatibility)
    
    # If expert_weights not provided, use uniform weights (backward compatibility)
    if expert_weights is None:
        expert_weights = np.ones((num_motion, R)) / R
    
    ind = -1; min_ll = 1e9
    for k in range(num_motion):
        Sigma, W = kernel_params_all_motions[k]
        expert_losses = []
        expert_predictions = []
        
        # Compute predictions/losses for all R experts
        for r in range(R):
            X_m_r = sigmoid(X_m @ Z[r])
            if mode == 'simple':
                Y = np.interp(X_m_r, X_test, Y_test)
                if isinstance(mu_ms[k], list):
                    ll = ((mu_ms[k][r]-Y).T)@(mu_ms[k][r]-Y)
                else:
                    ll = ((mu_ms[k]-Y).T)@(mu_ms[k]-Y)
                expert_losses.append(ll)
                expert_predictions.append(None)  # Not used in simple mode
            elif mode == 'variational':
                K_mm = spectral_kernel(X_m_r, X_m_r, Sigma, W) + jitter(X_m_r.shape[0])
                K_mn = spectral_kernel(X_m_r, X_test, Sigma, W)
                trace_avg_all_comps = jnp.sum(W**2)
                y_n_k = Y_test.reshape(-1, 1) # shape (n, 1)
                ll = elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y=0.1)
                expert_losses.append(ll)
                expert_predictions.append(None)  # Not used in variational mode
            elif mode == 'dt':
                if isinstance(mu_ms[k], list):
                    mean, _ = q(X_test, X_m_r, kernel_params_all_motions[k], mu_ms[k][r], A_ms[k][r], K_mm_invs[k][r])
                else:
                    mean, _ = q(X_test, X_m_r, kernel_params_all_motions[k], mu_ms[k], A_ms[k], K_mm_invs[k])
                ll = ((mean-Y_test).T)@(mean-Y_test)
                expert_losses.append(ll)
                expert_predictions.append(mean)
        
        # Use learned static weights for this class
        expert_losses = np.array(expert_losses)
        weights = expert_weights[k]  # Shape (R,)
        
        # Compute weighted loss using learned static weights
        weighted_ll = np.sum(weights * expert_losses)
        
        if ind == -1:
            ind = k; min_ll = weighted_ll
        elif min_ll > weighted_ll: 
            ind = k; min_ll = weighted_ll
    
    return ind