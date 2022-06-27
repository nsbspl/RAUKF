# Script for the plotting functions (including Figures for paper)
import numpy as np
from scipy.stats import norm
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from copy import copy
import types
import pickle
import os
sns.set(font_scale=2, rc={'text.usetex' : True, 'font.family': 'Helvetica', "font.size":24,"axes.titlesize":24,"axes.labelsize":20}, style="ticks")


def shiftCol(c,f):
    """Helper function to modify colormap"""

    nc = c*f
    return nc if nc<=1 else 1

def fixDig(self):
    """Helper function to enforce correct scientific notation"""

    self.format = '%.2f'
    if self._usetex or self._useMathText:
        self.format = r'$\mathdefault{%s}$' % self.format

def heatmap(fig, ax, data, name, NUM, xs, ys , xl, yl, cbl):
    """Single heatmap plot"""

    map = np.asarray(data).reshape(NUM,NUM)

    # Handle nans (color grey)
    nan_mask = np.isnan(map)
    map[nan_mask] = -1
    # outliers = np.quantile(map[~nan_mask], 0.75) + 1.5 * (np.quantile(map[~nan_mask], 0.75) - np.quantile(map[~nan_mask], 0.25))
    CN = 64
    cmap = copy(cm.get_cmap("rocket_r",256))
    rs = np.log10(np.logspace(1,4,CN))
    gs = np.log10(np.logspace(1,4,CN))
    bs = np.log10(np.logspace(1,4,CN))

    for i,c in enumerate(cmap.colors):
        j=i-(256-CN)
        if j>=0:
            cmap.colors[i][0] = shiftCol(c[0],rs[j])
            cmap.colors[i][1] = shiftCol(c[1],gs[j])
            cmap.colors[i][2] = shiftCol(c[2],bs[j])

    # heatmap = ax.imshow(map,origin='lower', vmax=0.04, vmin=0, cmap=cmap)
    heatmap = ax.imshow(map,origin='lower', cmap=cmap)
    cbar = fig.colorbar(heatmap,ax=ax, fraction=0.046, pad=0.08, extend="both")
    cbar.cmap.set_over('black') # outliers
    cbar.cmap.set_under('grey') # nan values
    cbar.ax.get_yaxis().labelpad = 25
    cbar.ax.set_ylabel(cbl, rotation=270)
    mf = tck.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((0, 0))
    mf._set_format = types.MethodType(fixDig,mf)

    cbar.ax.get_yaxis().set_major_formatter(mf)

    ax.set_xticks(np.arange(len(xs)))
    ax.set_yticks(np.arange(len(ys)))
    ax.set_xticklabels([f"{x:.1f}" for x in xs])
    ax.set_yticklabels([f"{y:.1f}" for y in ys])
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_title(f"{name}")

# Figure 2 and Figure 3
def heatmap_rmse(xlabl, ylabl, xs, ys, RMSEs):
    """State and parameter estimation heatmap based on RMSE"""

    fig, ax = plt.subplots(2,3, figsize=(16, 16))

    NUM = len(xs)
    # Dividing each RMSE by the range of the variable
    heatmap(fig,ax[0, 0], RMSEs[:, 0]/100, r"$V$", NUM, xs, ys, xlabl, ylabl, r"$\rm RMSE\:[mV]$")
    heatmap(fig,ax[0, 1], RMSEs[:, 1]/1, r"$w$", NUM, xs, ys, xlabl, ylabl, r"$\rm RMSE$")
    heatmap(fig,ax[1, 0], RMSEs[:, 2]/2, r"$g_{\rm leak}$", NUM, xs, ys, xlabl, ylabl, r"$\rm RMSE\:[\rm{mS/cm^{2}}]$")
    heatmap(fig,ax[1, 1], RMSEs[:, 3]/20, r"$g_{\rm slow}$", NUM, xs, ys, xlabl, ylabl, r"$\rm RMSE\:[\rm{mS/cm^{2}}]$")
    heatmap(fig,ax[1, 2], RMSEs[:, 4]/20, r"$g_{\rm fast}$", NUM, xs, ys, xlabl, ylabl, r"$\rm RMSE\:[\rm{mS/cm^{2}}]$")
    fig.delaxes(ax[0, 2])
    fig.tight_layout()
    return fig

def heatmap_tconv(xlabl, ylabl, xs, ys, CONVs):
    """Parameter estimation heatmap based on convergence time"""

    fig, ax = plt.subplots(1,3, figsize=(18, 18))

    NUM = len(xs)
    heatmap(fig,ax[0], CONVs[:, 0], r"$g_{\rm leak}$", NUM, xs, ys, xlabl, ylabl, r"$t_{conv}$")
    heatmap(fig,ax[1], CONVs[:, 1], r"$g_{\rm slow}$", NUM, xs, ys, xlabl, ylabl, r"$t_{conv}$")
    heatmap(fig,ax[2], CONVs[:, 2], r"$g_{\rm fast}$", NUM, xs, ys, xlabl, ylabl, r"$t_{conv}$")
    fig.tight_layout()
    return fig 

# Figure 1: noisy measurements vs t, groundtruth vs t; noisy input vs t
def plot_measurements_and_inputs(kf1):
    """
    Figure 1: Noisy measurements and noisy inputs
        - plot 1: y_{k, meas} vs t, v_gt vs t
        - plot 2: I_stim vs t 
    """
    t = kf1["t"]
    y = kf1["y"]
    I = kf1["I"]
    x_gt = kf1["x_gt"]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))

    ax[0].plot(t, y, color="orange", label=r"$y_{k, \rm meas}$")
    ax[0].plot(t, x_gt[:, 0], 'k--', linewidth=2, label=r"$V$")
    ax[0].set_ylabel(r"$V\,[\rm mV]$")
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))

    ax[1].plot(t, I, 'k', linewidth=2)
    ax[1].set_xlabel(r"\rm Time\, [\rm ms]")
    ax[1].set_ylabel(r"$I_{\rm stim}\,[\rm{\mu A/cm^{2}}]$")
    ax[1].set_yticks(np.arange(0, 150, 50))

    fig.tight_layout()

    return fig


# Figure 4: tracking of V, with groundtruth; tracking of W
# Figure 6: tracking of V, with groundtruth; tracking of W; subject to faulty measurements
# Figure 8: tracking of V, with groundtruth; tracking of W; subject to model mismatch
def plot_state_tracking(kf1, kf2=None, plot_y=False):
    """
    State tracking: RAUKF vs UKF
        - plot 1: v_raukf vs t, v_ukf_vs t, v_gt vs t
        - plot 2: w_raukf vs t, w_ukf vs t, w_gt vs t
    """
    t = kf1["t"]
    y = kf1["y"]
    I = kf1["I"]
    x_gt = kf1["x_gt"]
    x_kf1 = kf1["x"]
    error_kf1 = kf1["error"]
    name_kf1 = kf1["filter"].upper()

    # Error analysis
    N = len(t)
    # Compute RMSEs from N/2 --> N
    rmse_kf1 = np.sqrt(np.mean(error_kf1[int(N/2):] ** 2, axis=0))

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))

    if plot_y:
        ax[0].plot(t, y, color='orange', linewidth=2, label=r"$y_{k, \rm meas}$")
    ax[0].plot(t, x_kf1[:, 0], 'c-', linewidth=2, label=r"$\hat{{V}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[0].set_ylabel(r"$V\,[\rm mV]$")
    ax0 = ax[0].twinx()
    ax0.plot(np.NaN, np.NaN, 'c-', label=r"$\rm RMSE: {{{error}}}\:mV$".format(error=np.format_float_positional(rmse_kf1[0], precision=3)))

    ax[1].plot(t, x_kf1[:, 1], 'c-', linewidth=2, label=r"$\hat{{w}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[1].set_ylabel(r"$w$")
    ax[1].set_xlabel(r"\rm Time\, [\rm ms]")
    ax1 = ax[1].twinx()
    ax1.plot(np.NaN, np.NaN, 'c-', label=r"$\rm RMSE: {{{error}}}$".format(error=np.format_float_positional(rmse_kf1[1], precision=3)))

    if kf2 is not None:
        x_kf2 = kf2["x"]
        name_kf2  = kf2["filter"].upper()
        error_kf2 = kf2["error"]
        rmse_kf2 = np.sqrt(np.mean(error_kf2[int(N/2):] ** 2, axis=0))

        ax[0].plot(t, x_kf2[:, 0], 'g-', linewidth=2, label=r"$\hat{{V}}\,(\rm {{{name}}})$".format(name=name_kf2))
        ax0.plot(np.NaN, np.NaN, 'g-', label=r"$\rm RMSE: {{{error}}}\:mV$".format(error=np.format_float_positional(rmse_kf2[0], precision=3)))

        ax[1].plot(t, x_kf2[:, 1], 'g-', linewidth=2, label=r"$\hat{{w}}\,(\rm {{{name}}})$".format(name=name_kf2))
        ax1.plot(np.NaN, np.NaN, 'g-', label=r"$\rm RMSE: {{{error}}}$".format(error=np.format_float_positional(rmse_kf2[1], precision=3)))

    # FIXME: moved code around to fix legend ordering, but better way exists
    ax[0].plot(t, x_gt[:, 0], 'k--', linewidth=2, label=r"$V$")
    ax[1].plot(t, x_gt[:, 1], 'k--', linewidth=2, label=r"$w$")

    ax0.get_yaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax[0].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax0.legend(loc="lower left", bbox_to_anchor=(1.04, 0))
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax1.legend(loc="lower left", bbox_to_anchor=(1.04, 0))
    fig.tight_layout()
    
    return fig

# Error in V (with covariance), error in w (with covariance)
def plot_state_error(kf1, kf2=None):
    """
    State error + covariance tracking: RAUKF  vs UKF
    """
    t = kf1["t"]
    x_gt = kf1["x_gt"]
    error_kf1 = kf1["error"]
    std_kf1 = kf1["std"]
    name_kf1 = kf1["filter"].upper()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))

    ax[0].plot(t, error_kf1[:, 0], 'b-', label=r"$\hat{{V}} - V\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[0].fill_between(t, -3*std_kf1[:, 0], 3*std_kf1[:, 0], color='b', alpha=0.2, label=r"$\pm 3\sigma_{V}$")
    ax[0].set_ylabel(r"$V\,[\rm mV]$")

    ax[1].plot(t, error_kf1[:, 1], 'b-', label=r"$\hat{{w}} - w\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[1].fill_between(t, -3*std_kf1[:, 1], 3*std_kf1[:, 1], color='b', alpha=0.2, label=r"$\pm 3\sigma_{w}$")
    ax[1].set_xlabel(r"$\rm Time\, [\rm ms]$")
    ax[1].set_ylabel(r"$w$")

    if kf2 is not None:
        error_kf2 = kf2["error"]
        std_kf2 = kf2["std"]
        name_kf2 = kf2["filter"].upper()

        ax[0].plot(t, error_kf2[:, 0], 'g-', label=r"$\hat{{V}} - V\,(\rm {{{name}}})$".format(name=name_kf2))
        ax[0].fill_between(t, -3*std_kf2[:, 0], 3*std_kf2[:, 0], color='g', alpha=0.2, label=r"$\pm 3\sigma_{V}$")

        ax[1].plot(t, error_kf2[:, 1], 'g-', label=r"$\hat{{w}} - w\,(\rm {{{name}}})$".format(name=name_kf2))
        ax[1].fill_between(t, -3*std_kf2[:, 1], 3*std_kf2[:, 1], color='g', alpha=0.2, label=r"$\pm 3\sigma_{w}$")

    ax[0].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    fig.tight_layout()

    return fig

# Figure 5: tracking of parameters (3 conductances)
# Figure 7: tracking of parameters subject to faulty measurements
# Figure 9: tracking of parameters subject to model mismatch
def plot_parameter_tracking(kf1, kf2=None):
    """
    Parameter tracking: RAUKF vs UKF
        - plot 1: g_leak vs t
        - plot 2: g_slow vs t
        - plot 3: g_fast vs t
    """
    t = kf1["t"]
    x_gt = kf1["x_gt"]
    x_kf1 = kf1["x"]
    error_kf1 = kf1["error"]
    name_kf1 = kf1["filter"].upper()

    # Error analysis
    N = len(t)
    # Compute RMSEs from N/2 --> N
    rmse_kf1 = np.sqrt(np.mean(error_kf1[int(N/2):] ** 2, axis=0))

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 16))

    ax[0].plot(t, x_kf1[:, 2], 'b-', linewidth=2, label=r"$\hat{{g}}_{{\rm leak}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[0].set_ylabel(r"$g_{\rm leak} \,[\rm{mS/cm^{2}}]$")
    ax0 = ax[0].twinx()
    ax0.plot(np.NaN, np.NaN, 'b-', label=r"$\rm RMSE: {{{error}}}\:mS/cm^{{2}}$".format(error=np.format_float_positional(rmse_kf1[2], precision=3)))

    ax[1].plot(t, x_kf1[:, 3], 'b-', linewidth=2, label=r"$\hat{{g}}_{{\rm slow}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[1].set_ylabel(r"$g_{\rm slow}\,[\rm{mS/cm^{2}}]$")
    ax1 = ax[1].twinx()
    ax1.plot(np.NaN, np.NaN, 'b-', label=r"$\rm RMSE: {{{error}}}\:mS/cm^{{2}}$".format(error=np.format_float_positional(rmse_kf1[3], precision=3)))

    ax[2].plot(t, x_kf1[:, 4], 'b-', linewidth=2, label=r"$\hat{{g}}_{{\rm fast}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[2].set_xlabel(r"$\rm Time\, [ms]$")
    ax[2].set_ylabel(r"$g_{\rm fast}\,[\rm{mS/cm^{2}}]$")
    ax2 = ax[2].twinx()
    ax2.plot(np.NaN, np.NaN, 'b-', label=r"$\rm RMSE: {{{error}}}\:mS/cm^{{2}}$".format(error=np.format_float_positional(rmse_kf1[4], precision=3)))

    if kf2 is not None:
        x_kf2 = kf2["x"]
        error_kf2 = kf2["error"]
        name_kf2 = kf2["filter"].upper()
        rmse_kf2 = np.sqrt(np.mean(error_kf2[int(N/2):] ** 2, axis=0))

        ax[0].plot(t, x_kf2[:, 2], 'g-', linewidth=2, label=r"$\hat{{g}}_{{\rm leak}}\,(\rm {{{name}}})$".format(name=name_kf2))
        
        ax0.plot(np.NaN, np.NaN, 'g-', label=r"$\rm RMSE: {{{error}}}\:mS/cm^{{2}}$".format(error=np.format_float_positional(rmse_kf2[2], precision=3)))
        ax[1].plot(t, x_kf2[:, 3], 'g-', linewidth=2, label=r"$\hat{{g}}_{{\rm slow}}\,(\rm {{{name}}})$".format(name=name_kf2))
        
        ax1.plot(np.NaN, np.NaN, 'g-', label=r"$\rm RMSE: {{{error}}}\:mS/cm^{{2}}$".format(error=np.format_float_positional(rmse_kf2[3], precision=3)))
        ax[2].plot(t, x_kf2[:, 4], 'g-', linewidth=2, label=r"$\hat{{g}}_{{\rm fast}}\,(\rm {{{name}}})$".format(name=name_kf2))
        
        ax2.plot(np.NaN, np.NaN, 'g-', label=r"$\rm RMSE: N/A$")

    # FIXME: moved code around to fix legend ordering, but better way exists
    ax[0].plot(t, x_gt[:, 2], 'k--', linewidth=1, label=r"$g_{{\rm leak}} = {{{conductance}}}\:\rm mS/cm^{{2}}$".format(conductance=x_gt[0, 2]))
    ax[1].plot(t, x_gt[:, 3], 'k--', linewidth=1, label=r"$g_{{\rm slow}} = {{{conductance}}}\:\rm mS/cm^{{2}}$".format(conductance=x_gt[0, 3]))
    ax[2].plot(t, x_gt[:, 4], 'k--', linewidth=1, label=r"$g_{{\rm fast}} = {{{conductance}}}\:\rm mS/cm^{{2}}$".format(conductance=x_gt[0, 4]))
    ax0.get_yaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)        

    ax[0].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax0.legend(loc="lower left", bbox_to_anchor=(1.04, -0.1))

    ax[1].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax1.legend(loc="lower left", bbox_to_anchor=(1.04, -0.1))

    ax[2].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax2.legend(loc="lower left", bbox_to_anchor=(1.04, -0.1))
    fig.tight_layout()

    return fig

# Error in parameters (with covariance)
def plot_parameter_error(kf1, kf2=None):
    """
    Parameter error + covariance tracking: RAUKF  vs UKF
    """
    t = kf1["t"]
    x_gt = kf1["x_gt"]
    error_kf1 = kf1["x"] - x_gt
    std_kf1 = kf1["std"]
    name_kf1 = kf1["filter"].upper()

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 16))
    
    ax[0].plot(t, error_kf1[:, 2], 'b-', label=r"$\hat{{g}}_{{\rm leak}} - g_{{\rm leak}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[0].fill_between(t, -3*std_kf1[:, 2], 3*std_kf1[:, 2], color='b', alpha=0.2, label=r"$\pm 3\sigma_{g_{\rm leak}}$")
    ax[0].set_ylabel(r"$g_{\rm leak}\,\rm{mS/cm^{2}}$")

    ax[1].plot(t, error_kf1[:, 3], 'b-', label=r"$\hat{{g}}_{{\rm slow}} - g_{{\rm slow}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[1].fill_between(t, -3*std_kf1[:, 3], 3*std_kf1[:, 3], color='b', alpha=0.2, label=r"$\pm 3\sigma_{g_{\rm slow}}$")
    ax[1].set_ylabel(r"$g_{\rm slow}\,\rm{mS/cm^{2}}$")

    ax[2].plot(t, error_kf1[:, 4], 'b-', label=r"$\hat{{g}}_{{\rm fast}} - g_{{\rm fast}}\,(\rm {{{name}}})$".format(name=name_kf1))
    ax[2].fill_between(t, -3*std_kf1[:, 4], 3*std_kf1[:, 4], color='b', alpha=0.2, label=r"$\pm 3\sigma_{g_{\rm fast}}$")
    ax[2].set_xlabel(r"$\rm Time\, [\rm ms]$")
    ax[2].set_ylabel(r"$g_{\rm fast}\,\rm{mS/cm^{2}}$")

    if kf2 is not None:
        error_kf2 = kf2["x"] - x_gt
        std_kf2 = kf2["std"]
        name_kf2 = kf2["filter"].upper()

        ax[0].plot(t, error_kf2[:, 2], 'g-', label=r"$\hat{{g}}_{{\rm leak}} - g_{{\rm leak}}\,(\rm {{{name}}})$".format(name=name_kf2))
        ax[0].fill_between(t, -3*std_kf2[:, 2], 3*std_kf2[:, 2], color='g', alpha=0.2, label=r"$\pm 3\sigma_{g_{\rm leak}}$")

        ax[1].plot(t, error_kf2[:, 3], 'g-', label=r"$\hat{{g}}_{{\rm slow}} - g_{{\rm slow}}\,(\rm {{{name}}})$".format(name=name_kf2))
        ax[1].fill_between(t, -3*std_kf2[:, 3], 3*std_kf2[:, 3], color='g', alpha=0.2, label=r"$\pm 3\sigma_{g_{\rm slow}}$")

        ax[2].plot(t, error_kf2[:, 4], 'g-', label=r"$\hat{{g}}_{{\rm fast}} - g_{{\rm fast}}\,(\rm {{{name}}})$".format(name=name_kf2))
        ax[2].fill_between(t, -3*std_kf2[:, 4], 3*std_kf2[:, 4], color='g', alpha=0.2, label=r"$\pm 3\sigma_{g_{\rm fast}}$")

    ax[0].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    fig.tight_layout()

    return fig



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default="sim_ml_2d", help="Name of simulation")
    parser.add_argument("--sweep", type=str, default="", help="Name of parameter sweep")
    args, usr_args = parser.parse_known_args()

    parent_path = Path(__file__).parent.resolve()
    if args.sweep:
        # Load sweep data from results directory
        filename = args.sweep + ".pickle"
        outdir = os.path.join(parent_path, "results/sweep", filename)
        with open(outdir, 'rb') as fid:
            data = pickle.load(fid)

        NUM = len(data)
        X_dim = len(data[0]['states'][0])
        theta_dim = 3
        RMSEs = np.zeros((NUM, X_dim))

        for i, res in enumerate(data):
            RMSEs[[i]] = res["RMSE"]

        if args.sweep.startswith("ld"):
            # xs = np.linspace(0+1e-5,1-1e-5,int(NUM**0.5)+1)[1:]
            xs = np.linspace(0.1,0.5,int(NUM**0.5))
            xlabel = r"$\delta_0$"
            ylabel = r"$\lambda_0$"
        else:
            xs = np.linspace(1,10,int(NUM**0.5))
            xlabel = r"$b$"
            ylabel = r"$a$"
        ys = xs

    else:
        # Load simulation data from results directory
        outdir = os.path.join(parent_path, "results", args.sim)
        filters = os.listdir(outdir)
        datakeys = ["t", "x", "x_gt", "error", "std", "y", "I"]
        datadirs = []
        data = dict()

        for f in filters:
            data[f] = dict.fromkeys(datakeys)
            data[f]["filter"] = f
            datadir = os.path.join(outdir, f)

            for k in datakeys:
                filename = k + ".npy"
                filedir = os.path.join(datadir, filename)
                data[f][k] = np.load(filedir)
