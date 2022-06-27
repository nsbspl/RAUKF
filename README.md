# Robust adaptive neuronal state estimation 

Robust adaptive nonlinear Kalman filter for joint state and parameter estimation of single neuron models

## Setup

To run the simulation with the correct dependencies, it is easiest to set up a python3 virtual environment in the cloned repository:

```console
cd RAUKF
virtualenv -p /usr/bin/python3.8 venv     # create virtual env
echo 'export PYTHONPATH="${PYTHONPATH}:<absolute path to>/RAUKF"' >> venv/bin/activate
source venv/bin/activate                # activate virtual env
pip install -r requirements.txt         # install python dependencies
```

## Simulations

### Figures 1, 4, 5:
```console
python sim_ml_2d.py --filter ukf|ekf --input_noise oup [--robust] --sim sim_ml_2d
```

Example: Figures 4, 5
```console
# Simulations
python sim_ml_2d.py --filter ukf --input_noise oup --sim sim_ml_2d
python sim_ml_2d.py --filter ukf --robust --input_noise oup --sim sim_ml_2d

# Plotting
python -i plots.py --sim sim_ml_2d
>>> plot_state_tracking(data["raukf"], data["ukf"]); plt.show()
>>> plot_parameter_tracking(data["raukf"], data["ukf"]); plt.show()
```

### Figures 6, 7:
```console
python sim_ml_2d.py --filter ukf --input_noise oup --y_fault uni --robust --sim sim_ml_2d_faulty
```

### Figures 8, 9:
```console
python sim_ml_3d.py --filter ukf --input_noise oup [--robust] --sim sim_ml_3d_mismatch
```

### Figures 2, 3:
```console
python -i plots.py --sweep ab|ld
```