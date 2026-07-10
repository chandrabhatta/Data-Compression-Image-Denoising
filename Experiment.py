from functions import load_grayscale_image, add_gaussian_noise, sample_patches, ksvd, sparse_coding, reconstruct_image, psnr, remove_patch_means
import matplotlib.pyplot as plt
import numpy as np

# Load the images:
img1 = load_grayscale_image("Images/image11.jpg").astype(np.float64)
img2 = load_grayscale_image("Images/image2.jpg").astype(np.float64)
img3 = load_grayscale_image("Images/image3.jpg").astype(np.float64)

GothicDelft = load_grayscale_image("Images/GothicDelft.jpeg").astype(np.float64)



# Sample 6000 8 cross 8 patches of the image

# Extract training patches
Y_train, locations = sample_patches(GothicDelft,patch_size=8,K=6000,seed=0)
print(Y_train.shape)

# Processed image patches

Y_train, means = remove_patch_means(Y_train)

'''
# Reconstruct the image from the denoised patches:

reconstruct = reconstruct_image(Y_train, locations, image_shape=noisy_25.shape, patch_size=8)

# Reconstruct and plot the image

plt.subplot(1,3,3)
plt.imshow(reconstruct, cmap="gray")
plt.title("Reconstructed Image")
plt.show()

'''

# Perform K-SVD
n_atoms = 256
sparsity = 8
n_iter = 25

# Learn the dictionary:

D,X = ksvd(Y_train, n_atoms=n_atoms, sparsity = sparsity, n_iter=n_iter)

print(D.shape)
print(X.shape)

# Sparse code the noisy patches with the learned dictionary

Y_all, all_locations = sample_patches(GothicDelft, patch_size=8,K=None)
Y_all, all_means = remove_patch_means(Y_all)
X = sparse_coding(Y_all, D, sparsity, sigma=25)

# Perform denoising:
Y_denoised = D @ X

Y_denoised += all_means

# Reconstruct the image from the denoised patches:

denoised_image = reconstruct_image(Y_denoised, all_locations, image_shape=noisy_25.shape, patch_size=8)

# Compute the error

print("Noisy PSNR:",psnr(GothicDelft, GothicDelft))

print("Denoised PSNR:",psnr(GothicDelft, denoised_image))


# Reconstruct and plot the image

plt.figure(figsize=(15,5))

# Plot the Image:
plt.subplot(1,2,1)
plt.imshow(GothicDelft, cmap="gray")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(denoised_image, cmap="gray")
plt.title("K-SVD Denoised")

plt.show()


