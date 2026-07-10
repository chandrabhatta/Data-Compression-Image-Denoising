from PIL import Image
import numpy as np

# ------ Load image and add Gaussian noise -----------#

def load_grayscale_image(path):
    img = Image.open(path).convert("L") # converting to grayscale
    return np.array(img)

def add_gaussian_noise(image, sigma):

    image = image.astype(np.float32)
    noise = np.random.normal(loc=0.0,scale=sigma,size=image.shape)

    noisy = image + noise

    noisy_image = np.clip(noisy, 0, 255)

    return noisy_image.astype(np.uint8)

#------Sample patches and preprocess the image-------#

def sample_patches(image, patch_size = 8, K = 6000, seed = None):

    H, W = image.shape # Store the shape of the original image
    p = patch_size # Patch size 

    # All possible top-left coordinates:
    locations = [(i,j) for i in range(H - p + 1) for j in range(W - p + 1)]

    rng = np.random.default_rng(seed)

    # Randomly choose K locations:

    if K is None:
        chosen = np.arange(len(locations))
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(locations),size=K,replace=False)
    K = len(chosen)
    # Measurement matrix:
    Y = np.zeros((p*p, K))

    sampled_locations = []

    for k, idx in enumerate(chosen):

        [i,j] = locations[idx]

        patch = image[i:i+p, j:j+p]

        Y[:,k] = patch.flatten()

        sampled_locations.append((i,j))

    return Y, sampled_locations

def remove_patch_means(Y):

    means = np.mean(Y, axis=0)

    Y_zero_mean = Y - means

    return Y_zero_mean, means

# ----- K-SVD Algorithm -----------------------------#

def ksvd(Y, n_atoms, sparsity, n_iter):
    """
    Learn a dictionary using the K-SVD algorithm.
    """

    D = initialize_dictionary(Y, n_atoms)

    for iteration in range(n_iter):

        # Sparse coding step
        X = sparse_coding(Y, D, sparsity)

        # Dictionary update step
        for j in range(n_atoms):
            D, X = update_atom(Y, D, X, j)

        error = np.linalg.norm(Y - D @ X )/np.linalg.norm(Y)
        #print(f"error {error:.4f}")
        print(f"Iteration {iteration+1}/{n_iter} complete")

    return D, X

def initialize_dictionary(Y, n_atoms, seed = None):
    rng = np.random.default_rng(seed)

    # Number of training samples:
    N = Y.shape[1]

    if n_atoms>N:
        raise ValueError("Number of atoms cannot exceed number of training patches.")
    
    # Randomly choose columns from Y:

    indices = rng.choice(N,size= n_atoms,replace = False)

    D = Y[:,indices].copy()

    # Normalize each atom:

    norms = np.linalg.norm(D, axis=0)

    D/= norms

    return D

def sparse_coding(Y, D, sparsity, sigma=None):
    N = Y.shape[1]
    n_atoms = D.shape[1]

    X = np.zeros((n_atoms,N))

    for i in range(N):
        X[:,i] = omp(D, Y[:,i], sparsity, sigma=sigma)
    
    return X

def omp(D,y,s, sigma=None, C=1.15):

    y = y.ravel()
    n_atoms = D.shape[1]
    res = y.copy()
    supp = []
    x_supp = None

    for _ in range(s):

        correlations = D.T @ res # calculate the correlations
        correlations[supp] = 0
        k = np.argmax(np.abs(correlations))

        if k in supp:
            break
        supp.append(k)

        D_selected = D[:,supp]
        x_supp = np.linalg.lstsq(D_selected, y, rcond = None)[0]
        res = y - D_selected @ x_supp

        # Residual stopping
        if sigma is not None:
            if np.linalg.norm(res)**2 <= C * len(y) * sigma**2:
                break

    #print("Selected atoms:", supp)
    x = np.zeros(n_atoms)
    x[supp] = x_supp
    return x

def update_atom(Y,D,X,j):

    omega = np.nonzero(X[j,:])[0]
    
    if len(omega) == 0:
        return D,X
    
    # Restricted Residual:
    E = (Y[:,omega] - D @ X[:,omega] + np.outer(D[:,j], X[j,omega]))

    # Rank 1 approximation:
    U, S, Vt = np.linalg.svd(E, full_matrices=False)

    # Update dictionary atom:
    D[:,j] = U[:,0]

    # Update sparse coefficients:
    X[j,omega] = S[0]*Vt[0,:]

    return D,X

# ----- DL-SBL Algorithm --------------------------#

# DL-SBL AM

def dl_sbl_am(Y, n_atoms, sigma, n_iter=10,tol=1e-3):

    K = Y.shape[1]

    D = initialize_dictionary(Y, n_atoms, seed=None) 

    gamma = initialize_gamma(Y, n_atoms)

    for i in range(n_iter):

        Mu, Sigma_list = e_step(Y,D,gamma,sigma)

        gamma = update_gamma(Mu, Sigma_list)

        D_prev = D.copy()
        gamma_prev = gamma.copy()
    
        D = update_dictionary_am(Y,D,Mu,Sigma_list)

        #print(f"Iteration {i+1}: "f"Dictionary change = {np.linalg.norm(D - D_prev, ord='fro'):.6f}")

        #Y_hat = D @ Mu
        #err = np.linalg.norm(Y - Y_hat) / np.linalg.norm(Y)

        #print(f"Iter {i+1}: {err:.6f}")

        Mu, Sigma_list = e_step(Y, D, gamma, sigma) 


        dictionary_change = np.linalg.norm(D - D_prev, ord="fro")
        
        
        gamma_change = 0
        for k in range(K):
            gamma_change += np.linalg.norm(gamma[:,k] - gamma_prev[:,k])
        total_change = dictionary_change + gamma_change

        if total_change < tol:
            print("Converged")
            break
    
    return D, Mu

def initialize_gamma(Y,n_atoms):

    patch_energy = np.var(Y,axis=0)
    gamma = np.tile(patch_energy,(n_atoms,1))# unbiased prior for every coefficient

    return gamma

def e_step(Y,D,gamma,sigma):

    m,K = Y.shape
    n_atoms = D.shape[1]

    Mu = np.zeros((n_atoms,K))
    Sigma_list = []

    I = np.eye(m)

    for k in range(K):

        Gamma = np.diag(gamma[:,k])

        Phi = np.linalg.inv(sigma**2 * I + D @ Gamma @ D.T)

        Sigma = (Gamma - Gamma @ D.T @ Phi @ D @ Gamma)
        mu = (Sigma @ D.T @ Y[:,k])/sigma**2

        Mu[:,k] = mu

        Sigma_list.append(Sigma)
        
    return Mu, Sigma_list

def update_gamma(Mu, Sigma_List):

    n_atoms, K = Mu.shape

    gamma = np.zeros((n_atoms, K))

    for k in range(K):
        gamma[:,k] = Mu[:,k]**2 + np.diag(Sigma_List[k])
    return gamma

def update_dictionary_am(Y,D,Mu,Sigma_list,tol=1e-6,max_iter=20):

    m,K = Y.shape

    
    n_atoms = D.shape[1]

    M = Mu

    #print("Mu norm:", np.linalg.norm(Mu))

    Sigma_total = np.zeros((n_atoms,n_atoms))

    for k in range(K):
        mu = Mu[:,k]
        Sigma_total += Sigma_list[k] + np.outer(mu,mu)

    YMt = Y @ M.T

    D_old = D.copy()

    #Y_old = D_old @ Mu
    #err_old = np.linalg.norm(Y - Y_old) / np.linalg.norm(Y)

    #print("Before dictionary update:", err_old)

    for iteration in range(max_iter):

        D_new = D_old.copy()

        for i in range(n_atoms):
            v = YMt[:,i].copy()
            
            for j in range(i):
                v-= Sigma_total[i,j] * D_new[:,j]
            for j in range(i+1,n_atoms):
                v -= Sigma_total[i,j] * D_old[:,j]

            norm = np.linalg.norm(v)

            if norm > 1e-12:
                D_new[:,i] = v/norm

        #Y_new = D_new @ Mu
        #err_new = np.linalg.norm(Y - Y_new) / np.linalg.norm(Y)

        #print("After dictionary update:", err_new)


        if np.linalg.norm(D_new - D_old, ord = "fro")<tol:
            break

        D_old = D_new

    #print("Sigma diag:", np.diag(Sigma_total)[:5])
    #print("YM norm:", np.linalg.norm(YMt))
    #print("Atom norms:", np.linalg.norm(D_new,axis=0)[:5])
    #print("Sigma symmetry error:",np.linalg.norm(Sigma_total-Sigma_total.T))

    #print("Mu norm:", np.linalg.norm(Mu))
    return D_new

# DL-SBL ALS

def update_dictionary_als(Y, D, Mu, Sigma_list,tol=1e-6,max_iter=50,alpha=1e-4,beta=0.5):

    m, K = Y.shape
    n_atoms = D.shape[1]

    # M = [mu1 ... muK]
    M = Mu

    # Sigma = Σ_k (Sigma_k + mu_k mu_k^T)
    Sigma_total = np.zeros((n_atoms, n_atoms))

    for k in range(K):
        mu = Mu[:, k]
        Sigma_total += Sigma_list[k] + np.outer(mu, mu)

    # YM^T
    YMt = Y @ M.T

    D_old = D.copy()

    for iteration in range(max_iter):

        # Gradient of the objective
        G = -2 * YMt + 2 * D_old @ Sigma_total

        # Current objective
        f_old = (np.trace(D_old @ Sigma_total @ D_old.T)- 2 * np.trace(YMt.T @ D_old))

        eta = 1.0

        # Armijo backtracking
        while True:

            D_trial = D_old - eta * G

            # Normalize columns
            norms = np.linalg.norm(D_trial, axis=0)
            norms[norms == 0] = 1
            D_trial /= norms

            f_trial = (np.trace(D_trial @ Sigma_total @ D_trial.T)- 2 * np.trace(YMt.T @ D_trial))

            if f_trial <= f_old - alpha * eta * np.linalg.norm(G, "fro")**2:
                break

            eta *= beta

            if eta < 1e-12:
                break

        if np.linalg.norm(D_trial - D_old, ord="fro") < tol:
            D_old = D_trial
            break

        D_old = D_trial

    return D_old

# Reconstruction of Image and SNR
def reconstruct_image(Y_denoised, locations, image_shape, patch_size=8):

    # Initialize the image and the weights
    image = np.zeros(image_shape, dtype=float)
    weight = np.zeros(image_shape, dtype=float)

    for k, (r,c) in enumerate(locations):
        patch_hat = Y_denoised[:,k].reshape((patch_size,patch_size))
        image[r:r+patch_size, c:c+patch_size] +=patch_hat
        weight[r:r+patch_size, c:c+patch_size] += 1

    # Average the overlapping patches
    weight[weight == 0] = 1
    image/=weight

    return image

def psnr(original, reconstructed):

    mse = np.mean((original-reconstructed)**2)

    if mse == 0:
        return np.inf
    
    return 20*np.log10(255/np.sqrt(mse))

