# Joint state and parameter estimation for 2D Morris-Lecar model
import numpy as np
import yaml
import argparse
from pathlib import Path
import os

from models.morris_lecar import MorrisLecar2D
from models.utils import input_current, simulate_measurement
from filters.ukf import Ukf
from filters.ekf import Ekf

# Simulations to reproduce figures:
#   Figures 1, 4, 5:
#   python sim_ml_2d.py --filter ukf|ekf --input_noise oup [--robust] --sim sim_ml_2d
#
#   Figures 6, 7:
#   python sim_ml_2d.py --filter ukf --input_noise oup --y_fault uni --robust --sim sim_ml_2d_faulty


# Fix random seed to get reproducible results
np.random.seed(42)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--filter", type=str, choices=["ukf", "ekf"], default="ukf", help="Nonlinear Kalman filter implementation used for estimation")
parser.add_argument("--robust", action="store_true", help="Activate covariance adaptation")
parser.add_argument("--input_noise", type=str, choices=["oup", "wgn"], default="oup", help="The type of noisy input to inject")
parser.add_argument("--y_fault", type=str, choices=["rnd", "uni", "mi"], default="", help="The type of measurement fault to simulate")
parser.add_argument("--freq", type=float, default=0.5, help="Fault frequency, f \in (0, 1]")
parser.add_argument("--step", nargs="+", type=int, help="Step input ON/OFF time(s) [ms]")
parser.add_argument("--sim", type=str, default=str(os.path.basename(__file__).split(".")[0]), help="Name of simulation for results classification")
args, usr_args = parser.parse_known_args()

# Load simulation parameters
parent_path = Path(__file__).parent.resolve()
parameter_path = os.path.join(parent_path, "../models/morris_lecar_parameters.yml")
with open(parameter_path, "r") as fid:
    try:
        cfg = yaml.safe_load(fid)
    except yaml.YAMLError as exc:
        print(exc)

# Simulation setup
p = cfg["2d"] # model parameters
N_sim = int(p["t_stop"] / p["dt"])
tt = np.arange(0, p["t_stop"], p["dt"])

# Model
I_inj, I_inj_noisy = input_current(p, **vars(args))
ml_neuron = MorrisLecar2D(p)
x_gt = ml_neuron.test(I_inj_noisy, int_factor=10) # groundtruth data

# Measurements (with optional artificial faults)
out_membrane, y_faulty, fault_mask = simulate_measurement(x_gt, p, **vars(args))
if args.y_fault:
    out_membrane = y_faulty

# Play around with initial conditions
# I.C.: v0=0, w0=0, g_leak=10, g_slow=80, g_fast=140
theta_keys = ["g_leak", "g_slow", "g_fast"] # parameters to estimate

theta = [list(p.keys()).index(param) for param in theta_keys]
x0 = np.array([[-100.0, 0.5, 10.0, 80, 140]])
P0  = np.diag(np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4]))
Q0 = np.diag(np.array([1e1, 1e-3, 1e1, 1e1, 1e1]))
R0 = 0.1 * np.diag(np.array([p["obs_noise"]]))

# Recursive state estimation
if args.filter == 'ukf':
    kf = Ukf(ml_neuron, out_membrane, I_inj_noisy, theta, x0, P0, Q0, R0, kappa=0, sigma=0.5, robust=args.robust)
    x, P = kf.run_estimation(resample=True, int_factor=10)

elif args.filter == 'ekf':
    kf = Ekf(ml_neuron, out_membrane, I_inj_noisy, theta, x0, P0, Q0, R0, sigma=0.5, robust=args.robust)
    x, P = kf.run_estimation()

# Check number of covariance corrections
if args.robust:
    print("Number of robust covariance updates: {}".format(len(np.where(kf.phi > kf.threshold)[0])))
    args.filter = "ra" + args.filter

# Error analysis
std = np.array([np.sqrt(np.diag(P[i:i+P.shape[1]])) for i in range(0, P.shape[0], P.shape[1])])
for param in theta_keys:
    x_gt = np.concatenate((x_gt, p[param] * np.ones((N_sim, 1))), axis=1)

error = x - x_gt

# Saving results
outdir = os.path.join(parent_path, "results", args.sim, args.filter)
print("Results written to: {}".format(outdir))
Path(outdir).mkdir(parents=True, exist_ok=True)

outfile = os.path.join(outdir, "t.npy")
np.save(outfile, tt)

outfile = os.path.join(outdir, "x.npy")
np.save(outfile, x)

outfile = os.path.join(outdir, "x_gt.npy")
np.save(outfile, x_gt)

outfile = os.path.join(outdir, "error.npy")
np.save(outfile, error)

outfile = os.path.join(outdir, "std.npy")
np.save(outfile, std)

outfile = os.path.join(outdir, "y.npy")
np.save(outfile, out_membrane)

outfile = os.path.join(outdir, "I.npy")
np.save(outfile, I_inj_noisy)
