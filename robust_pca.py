import numpy as np
from pathlib import Path
from scipy.sparse import random as sparse_random
from PIL import Image
import matplotlib.pyplot as plt

import proj_grad

def generate_E0(m, n, density, low=-10, high=10, seed=None):
    """
    Generate an m x n sparse matrix with:
    - support (positions of non-zeros) chosen uniformly at random
    - non-zero entries drawn i.i.d. Uniform[low, high]

    Parameters
    ----------
    m, n     : matrix dimensions
    density  : fraction of entries that are non-zero (0 < density <= 1)
    low, high: range for the non-zero values
    seed     : optional random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # scipy's sparse_random picks the support uniformly at random,
    # and we override the values with our own uniform distribution
    E0 = sparse_random(
        m, n,
        density=density,
        random_state=rng,
        data_rvs=lambda size: rng.uniform(low, high, size=size)
    )
    return E0

def proj_M(X,k):
    """Project X onto the set with sparsity <=k"""
    # Flatten the matrix and get the indices of the k largest absolute values
    flat_X = X.flatten()
    if k == 0:
        return np.zeros_like(X)  # If k is 0, return a zero matrix

    # Get the indices of the k largest absolute values
    idx = np.argpartition(np.abs(flat_X), -k)[-k:]

    # Create a new matrix with zeros and set the k largest entries
    proj_X = np.zeros_like(flat_X)
    proj_X[idx] = flat_X[idx]

    # Reshape back to the original matrix shape
    return proj_X.reshape(X.shape)

def proj_M2(X0,k):
    """Project X onto a sparsity constraint using hard thresholding. This is a more efficient implementation than proj_M."""
    # Flatten the matrix and get the indices of the k largest absolute values
    X=X0.copy()
    thresh = 1
    X[np.abs(X) < thresh] = 0
    return X

def proj_D(X, rank):
    """Project X onto the set of matrices with rank <= rank"""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S[rank:] = 0  # Zero out all but the top 'rank' singular values
    return U @ np.diag(S) @ Vt

def evaluate_recovery(L, E0, res):
    rank_L0 = np.linalg.matrix_rank(L)
    sparsity_E0 = np.count_nonzero(E0.toarray())
    print(f"Original low-rank matrix L has rank {rank_L0}.")
    print(f"Original sparse matrix E0 has {sparsity_E0} non-zero entries.")
    
    rel_error = np.linalg.norm(res.x - L, 'fro') / np.linalg.norm(L, 'fro')
    rank_res = np.linalg.matrix_rank(res.x)
    E_res = Y - res.x
    sparsity_res = (np.abs(E_res) > 1e-4).sum()  # Count non-zero entries in E_res
    
    print(f"Recovered matrix L has rank {rank_res}, recovered matrix E has sparsity {sparsity_res}.")
    print(f"Relative error between recovered matrix and original low-rank matrix: {rel_error:.6f}")


def pca_synthetic_data(n, rank_max, density, seed=42):
    """
    Generate synthetic data for robust PCA.

    Parameters
    ----------
    n         : int
                Size of the square matrix (n x n)
    rank_max  : int
                Maximum rank of the low-rank matrix
    density   : float
                Density of the sparse matrix
    seed      : int
                Random seed for reproducibility
    """
    rng = np.random.default_rng(seed=seed)

    L_0 = rng.normal(loc=0, scale=1, size=(n, rank_max))

    L = np.dot(L_0, L_0.T)
    print(L.shape)
    rng = np.random.default_rng(seed=42)
    # E = sparse_random(n, n, density=0.1, format='csr', random_state=np.random.uniform(-500, 500))

    E0 = generate_E0(n, n, density, seed=42)
    print(E0.shape, E0.nnz)  

    Y = L + E0.toarray()

    k = int(np.ceil(density * n * n))  # Expected number of non-zero entries
    return L, E0, Y, k


def load_image_folder_to_matrix(path, image_size=None, max_images=None):
    """Load JPG images from a folder and stack flattened grayscale vectors into a matrix.

    Each image becomes one column in the returned matrix Y.
    """
    image_dir = Path(path)
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Image folder not found: {path}")

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_files:
        raise ValueError(f"No JPG or PNG images found in folder: {path}")

    if max_images is not None:
        image_files = image_files[:max_images]

    flattened_images = []
    target_shape = None
    for image_path in image_files:
        with Image.open(image_path) as img:
            img = img.convert("L")
            if image_size is not None:
                img = img.resize(image_size, resample=Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float64)
            if target_shape is None:
                target_shape = arr.shape
            elif arr.shape != target_shape:
                raise ValueError(
                    f"Image sizes differ: {image_path} has shape {arr.shape}, expected {target_shape}"
                )
            flattened_images.append(arr.ravel())

    Y = np.stack(flattened_images, axis=1)
    return Y, target_shape


def robust_pca_video(path, image_size=None, max_images=None):
    Y, image_shape = load_image_folder_to_matrix(path, image_size=image_size, max_images=max_images)
    print(f"Loaded {Y.shape[1]} images of shape {image_shape} into matrix Y with shape {Y.shape}.")
    return Y


#path = "dataset2014/dataset/baseline/PETS2006/input"
#path = "dataset2014/dataset/badWeather/skating/input"
path = "dataset2014/dataset/baseline/highway/input"

width = 320
height = 240

rank_max = 2
# Use grayscale image vectors as columns in Y.
Y = robust_pca_video(path, image_size=(width, height), max_images=400)

k = int(np.ceil(0.0001 * Y.size))
obj = lambda X: 0.5 * np.linalg.norm(Y - X - proj_M(Y - X, k), 'fro')**2
grad = lambda X: X - Y + proj_M(Y - X, k)
proj = lambda X: proj_D(X, rank_max)

seed = 1
np.random.seed(seed)
x_start = np.random.rand(*Y.shape)
#resBB = proj_grad.pgd_mon(x0=x_start, f=obj, grad=grad, proj=proj,
#                        max_iter=200, TOL=1e-4, barzilai_borwein=True)

resBB = proj_grad.pgd_avg(x0=x_start, f=obj, grad=grad, proj=proj,
                        max_iter=200, TOL=1e-4)

for i in range(rank_max):
    #img1 = resBB.x[:, i].reshape(height, width) / max(resBB.x[:, i])
    img1_scaled = np.clip(resBB.x[:, i].reshape(height, width), 0, 255)
    #img1_scaled = img1 * 255

    im_save = Image.fromarray((img1_scaled).astype(np.uint8))
    im_save.save(f"img/recovered_low_rank_matrix_{i}.png")

ref_idx = 272
ref_img = Y[:, ref_idx].reshape(height, width)
diff_img = np.abs(ref_img - img1_scaled)
foreground = np.clip(diff_img, 0, 255)
im_save_foreground = Image.fromarray((foreground).astype(np.uint8))
im_save_foreground.save(f"img/foreground_{ref_idx}.png")

im_save_ref = Image.fromarray((ref_img).astype(np.uint8))
im_save_ref.save(f"img/reference_image_{ref_idx}.png")

print(resBB)

plt.imshow(img1_scaled, cmap='gray')
plt.title("Recovered Low-Rank Matrix")
plt.axis('off')
plt.show()


#res = proj_grad.pgd_mon(x0=x_start, f=obj, grad=grad, proj=proj,
#                        max_iter=1000, TOL=1e-4, barzilai_borwein=False)
#print(res)
