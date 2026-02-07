import glob, os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# -------- settings --------
snap_glob = "snapshots-morozov/*.h5"
field     = "Trc"   # Trc, vx, vy, omega_devmean, cxx, cxy, cyy, ...
tmax      = 10.0
out_dir   = f"frames_{field}_t10"
dpi       = 150
# --------------------------

os.makedirs(out_dir, exist_ok=True)
files = sorted(glob.glob(snap_glob))
if not files:
    raise RuntimeError(f"No files match {snap_glob}")

def get_xy(f):
    # hashed coordinate names
    x_key = [k for k in f["scales"].keys() if k.startswith("x_hash_")][0]
    y_key = [k for k in f["scales"].keys() if k.startswith("y_hash_")][0]
    x = f["scales"][x_key][...]
    y = f["scales"][y_key][...]
    return x, y

frame = 0

for fn in files:
    with h5py.File(fn, "r") as f:
        if field not in f["tasks"]:
            raise KeyError(f"{fn} has no tasks/{field}. Available: {list(f['tasks'].keys())}")

        A = f["tasks"][field][...]          # (nt, Nx, Ny)
        t = f["scales/sim_time"][...]       # (nt,)
        x, y = get_xy(f)

    # pick times <= tmax and finite frames
    keep = np.where(t <= tmax)[0]
    if keep.size == 0:
        continue

    # pcolormesh expects 2D mesh; note Dedalus stores (Nx,Ny), so we transpose for (Ny,Nx)
    X, Y = np.meshgrid(x, y, indexing="xy")  # (Ny, Nx)

    for k in keep:
        Ak = A[k]  # (Nx, Ny)

        if not np.isfinite(Ak).all():
            # skip bad frames (you had NaNs later; t<=10 should be fine)
            continue

        plt.figure(figsize=(7, 3))
        plt.pcolormesh(X, Y, Ak.T, shading="auto")
        plt.colorbar()
        plt.title(f"{field}   t={t[k]:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        png = os.path.join(out_dir, f"frame_{frame:05d}.png")
        plt.savefig(png, dpi=dpi)
        plt.close()
        frame += 1

print(f"Wrote {frame} PNGs to {out_dir}/")
