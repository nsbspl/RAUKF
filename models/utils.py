# Utility functions for neuronal model simulations
import numpy as np

# Fix seed for reproducible results
np.random.seed(0)

def OU_process(params):
    """Ornstein-Uhlenbeck process"""

    N_sim = int(params["t_stop"] / params["dt"])
    N_tau = np.sqrt(2 / params["tau"])
    eta = np.zeros((N_sim, 1))
    x = np.random.randn(N_sim, 1)

    k = 0
    linspace = np.arange(params["dt"], params["t_stop"], params["dt"])
    for t in linspace:
        E_inf = params["tau"] * N_tau * x[k, 0] / np.sqrt(params["dt"])
        eta[k + 1, 0] = E_inf + (eta[k, 0] - E_inf) * np.exp(-params["dt"] / params["tau"])
        k = k + 1
    return eta, x

def input_current(params, **kwargs):
    """
    Helper provides different types of input currents for simulation:
        - white Gaussian noise
        - step current (+ WGN)   
        - Ornstein-Uhlenbeck process
    """
    N = int(params["t_stop"] / params["dt"])
    I = params["I_stim"] * np.ones((N, 1))

    input_type = kwargs["input_noise"]
    eta, wgn = OU_process(params)
    if input_type == "oup":
        I_noisy = I + params["I_noise"] * eta
    elif input_type == "wgn":
        I_noisy = I + params["I_noise"] * wgn
    else:
        I_noisy = I

    # Apply steps if requested and valid
    if kwargs["step"] is not None:
        t_start = 0
        t_end = N
        N_times = len(kwargs["step"])
        steps = (np.array(kwargs["step"]) / params["dt"]).astype(int)
        for t in range(N_times - 1, 0, -2):
            if steps[t] < t_end and steps[t] > t_start:
                I_noisy[steps[t]:t_end] = 0
                t_end = steps[t-1]
        I_noisy[0:steps[0]] = 0

    return I, I_noisy

def simulate_measurement(x_gt, params, **kwargs):
    """
    Helper simulates noisy measurements (with optional faults):
        Noise:
            - WGN
        Faults:
            - random faults
            - uniformly occurring faults
    """

    N = int(params["t_stop"] / params["dt"])
    y = x_gt[:, [params["obs_index"]]] + np.sqrt(params["obs_noise"]) * np.random.randn(N, 1)
    y_faulty  = y
    fault_type = kwargs["y_fault"]
    if fault_type == "rnd":
        # TODO: add start/stop for faults
        random_faults = np.random.choice(np.arange(2, N), size=int(N * kwargs["freq"]), replace=False)
        y_faulty[random_faults] = x_gt[:, [params["obs_index"]]][random_faults] + 2 * np.sqrt(params["obs_noise"]) * np.random.randn(len(random_faults), 1)
    elif fault_type == "uni":
        uni_faults = np.arange(int(N/4), int(3 * N / 4), 1)
        y_faulty[uni_faults] = x_gt[:, [params["obs_index"]]][uni_faults] + 5 * np.sqrt(params["obs_noise"]) * np.random.randn(len(uni_faults), 1)

    # Keep track of nominal data
    not_faulty_meas = ~np.isnan(y_faulty)[:, 0]
    
    return y, y_faulty, not_faulty_meas 