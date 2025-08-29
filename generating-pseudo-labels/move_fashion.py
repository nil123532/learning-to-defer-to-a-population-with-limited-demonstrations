import os
import shutil

# Define e (label counts) and p (expert strengths)
labels = [20, 40, 60, 100, 200, 500, 2500]
expert_strengths = [8]

for e in labels:
    for p in expert_strengths:
        src = os.path.join(f"FASHION/ex{p}_x{e}_seed0_attn", "ckp.latest")
        dst_dir = os.path.join("..", "..", "l2d-pop", "pretrained", "fashion", "attention", f"e_{e}_p{p}")
        dst = os.path.join(dst_dir, "ckp.latest")

        # Ensure destination directory exists
        os.makedirs(dst_dir, exist_ok=True)

        # Move if source file exists
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")
        else:
            print(f"WARNING: {src} not found. Skipping.")
