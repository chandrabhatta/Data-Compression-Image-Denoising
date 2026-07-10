from functions import (
    load_grayscale_image,
    add_gaussian_noise,
    sample_patches,
    ksvd,
    sparse_coding,
    reconstruct_image,
    psnr,
    remove_patch_means,
)

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Parameters
# --------------------------------------------------

patch_size = 8
K = 6000
n_atoms = 256
sparsity = 8
n_iter = 25

noise_levels = [5, 10, 15, 25]

# --------------------------------------------------
# Load images
# --------------------------------------------------

images = {
    "Image 1": load_grayscale_image("Images/image11.jpg").astype(np.float64),
    "Image 2": load_grayscale_image("Images/image2.jpg").astype(np.float64),
    "Image 3": load_grayscale_image("Images/image3.jpg").astype(np.float64),
}

# --------------------------------------------------
# Store results
# --------------------------------------------------

results = []

# --------------------------------------------------
# Main experiment
# --------------------------------------------------

for image_name, img in images.items():

    print(f"\n============================")
    print(image_name)
    print("============================")

    for sigma in noise_levels:

        print(f"\nNoise σ = {sigma}")

        # ------------------------------------------
        # Add Gaussian noise
        # ------------------------------------------

        noisy = add_gaussian_noise(img, sigma)

        # ------------------------------------------
        # Training patches
        # ------------------------------------------

        Y_train, locations = sample_patches(
            noisy,
            patch_size=patch_size,
            K=K,
            seed=0,
        )

        Y_train, means = remove_patch_means(Y_train)

        # ------------------------------------------
        # Learn dictionary
        # ------------------------------------------

        D, X_train = ksvd(
            Y_train,
            n_atoms=n_atoms,
            sparsity=sparsity,
            n_iter=n_iter,
        )

        # ------------------------------------------
        # Denoise ALL patches
        # ------------------------------------------

        Y_all, all_locations = sample_patches(
            noisy,
            patch_size=patch_size,
            K=None,
        )

        Y_all, all_means = remove_patch_means(Y_all)

        X = sparse_coding(
            Y_all,
            D,
            sparsity,
            sigma=sigma,
        )

        Y_denoised = D @ X

        Y_denoised += all_means

        denoised = reconstruct_image(
            Y_denoised,
            all_locations,
            image_shape=img.shape,
            patch_size=patch_size,
        )

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        noisy_psnr = psnr(img, noisy)
        denoised_psnr = psnr(img, denoised)

        improvement = denoised_psnr - noisy_psnr

        mse = np.mean((img - denoised) ** 2)

        results.append(
            {
                "Image": image_name,
                "Sigma": sigma,
                "Noisy PSNR": noisy_psnr,
                "Denoised PSNR": denoised_psnr,
                "Improvement": improvement,
                "MSE": mse,
            }
        )

        print(f"Noisy PSNR     : {noisy_psnr:.2f} dB")
        print(f"Denoised PSNR  : {denoised_psnr:.2f} dB")
        print(f"Improvement    : {improvement:.2f} dB")
        print(f"MSE            : {mse:.2f}")

        # ------------------------------------------
        # Display images
        # ------------------------------------------

        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(noisy, cmap="gray")
        plt.title(f"Noisy (σ={sigma})")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(denoised, cmap="gray")
        plt.title("Denoised")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

# --------------------------------------------------
# Print summary table
# --------------------------------------------------

print("\n==============================")
print("Summary")
print("==============================")

print(
    "{:<10} {:<8} {:<12} {:<15} {:<12} {:<12}".format(
        "Image",
        "Sigma",
        "Noisy PSNR",
        "Denoised PSNR",
        "Improve",
        "MSE",
    )
)

for r in results:

    print(
        "{:<10} {:<8} {:<12.2f} {:<15.2f} {:<12.2f} {:<12.2f}".format(
            r["Image"],
            r["Sigma"],
            r["Noisy PSNR"],
            r["Denoised PSNR"],
            r["Improvement"],
            r["MSE"],
        )
    )

plt.figure(figsize=(6,5))

for image_name in images.keys():

    sigma = []
    psnr_values = []

    for r in results:

        if r["Image"] == image_name:
            sigma.append(r["Sigma"])
            psnr_values.append(r["Denoised PSNR"])

    plt.plot(
        sigma,
        psnr_values,
        marker='o',
        linewidth=2,
        label=image_name
    )

plt.xlabel("Noise Standard Deviation (σ)")
plt.ylabel("Denoised PSNR (dB)")
plt.title("K-SVD Performance vs Noise Level")
plt.grid(True)
plt.legend()
plt.show()

