import os
import shutil

# Define base paths (relative to artificial_expert_labels)
src_base = os.path.join("cifar10")
dst_base = os.path.join("..", "..", "l2d-pop", "data", "generated_expert_labels_cifar")

# Define your label sizes and expert strengths
labels = [20, 40, 60, 100, 200, 500, 2500]
expert_strengths = [8]

for l in labels:
    for p in expert_strengths:
        src_folder = f"L_{l}_p{p}"
        dst_folder = f"e_{l}_p{p}"

        src_path = os.path.join(src_base, src_folder)
        dst_path = os.path.join(dst_base, dst_folder)

        # Make sure destination folder exists
        os.makedirs(dst_path, exist_ok=True)

        # Copy the .npy files
        for fname in ["train_array.npy", "test_array.npy"]:
            src_file = os.path.join(src_path, fname)
            dst_file = os.path.join(dst_path, fname)

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} -> {dst_file}")
            else:
                print(f"WARNING: {src_file} does not exist. Skipping.")
