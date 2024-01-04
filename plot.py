from huggingface_hub import HfFileSystem
import os
import matplotlib.pyplot as plt
import torch

# Read in example from Hugging Face as an example inputs
fs = HfFileSystem(token=os.environ["HF_TOKEN"])
with fs.open(f"datasets/jacobbieker/era5-42hour/data/0/00079fff-f2bf-4785-bc82-e0b921fff13d.pt", "rb") as f:
    data = torch.load(f)
x, y = data

x = x[0]
y = y[0]

variable_map = {0: "Temperature", 1: "U Wind", 2: "V Wind"}

# Plot 3 x 4 grid of inputs
fig, ax = plt.subplots(3, 4)
for i in range(3):
    for j in range(4):
        ax[i, j].imshow(x[i, j].cpu().numpy())
        # Turn off x and y ticks and labels
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_xlabel(f"Timestep {j}", fontsize=8)
        if j == 0:
            ax[i, j].set_ylabel(variable_map[i])
fig.suptitle("ERA5 input frames")
plt.savefig("era5_input.png")
plt.close()

# Subtract the value of the previous frame from the current frame, and plot it
fig, ax = plt.subplots(3, 4)
for i in range(3):
    for j in range(4):
        ax[i, j].imshow(x[i, j].cpu().numpy() - x[i, j - 1].cpu().numpy())
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_xlabel(f"Timestep {j}", fontsize=8)
        if j == 0:
            ax[i, j].set_ylabel(variable_map[i])
fig.suptitle("Difference between ERA5 input frames")
plt.savefig("era5_diff.png")
plt.close()

# Plot 3x7 grid of GT
fig, ax = plt.subplots(3, 7)
for i in range(3):
    for j in range(7):
        ax[i, j].imshow(y[i, j].cpu().numpy())
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_xlabel(f"Timestep {j}", fontsize=8)
        if j == 0:
            ax[i, j].set_ylabel(variable_map[i])
fig.suptitle("ERA5 target frames")
plt.savefig("era5_target.png")
plt.close()

# Now plot GOES and IMERG input example
with fs.open(f"datasets/jacobbieker/goes-imerg-42hour-test/data/0/521.pt", "rb") as f:
    data = torch.load(f)
x, y = data

x = x[0]
y = y[0]
# Plot 4 x 4 grid of the most recent GOES Inputs
fig, ax = plt.subplots(4, 4)
for i in range(16):
    ax[i // 4, i % 4].imshow(x[i, -1].cpu().numpy())
    ax[i // 4, i % 4].set_xticks([])
    ax[i // 4, i % 4].set_yticks([])
    ax[i // 4, i % 4].set_xlabel(f"Channel {i}", fontsize=8)
fig.suptitle("GOES input Channels")
plt.savefig("goes_input.png")
plt.close()
