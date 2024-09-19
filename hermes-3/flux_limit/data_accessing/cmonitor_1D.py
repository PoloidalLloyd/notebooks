#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import boutdata
from boututils.options import BOUTOptions
from boutdata.collect import create_cache
import argparse
import os
import time as tm


def cmonitor(path, save=False, plot=False, table=True, neutrals=False):
    """
    Produce convergence report of 1D Hermes-3 simulation.
    Plots of process conditions at a selected point.
    As well as solver performance indices.
    In addition to the plot, a CLI-friendly table can be produced.

    Inputs
    -----
    path: str, path to case directory
    save: bool, save figure. Saved by case name
    plot: bool, show plot
    table: bool, show table
    neutrals: bool, focus on neutrals in the plot
    """
    tstart = tm.time()

    if path == ".":
        casename = os.path.basename(os.getcwd())
    else:
        casename = os.path.basename(path)
    print(f"Reading {casename}")
    print("Calculating...", end="")

    # Reading with cache for extra speed
    cache = create_cache(path, "BOUT.dmp")
    print("..cache", end="")

    def get_var(name):
        return boutdata.collect(
            name,
            path=path,
            yguards="include_upper",  # Always with guards to minimise mistakes
            xguards=True,  # Always with guards to minimise mistakes
            strict=True,   # To prevent reading wrong variable by accident
            info=False,
            datafile_cache=cache
        ).squeeze()

    # Get normalisations and geometry
    Nnorm = get_var("Nnorm")
    Tnorm = get_var("Tnorm")
    Omega_ci = get_var("Omega_ci")
    dx = get_var("dx")
    dz = get_var("dz")
    J = get_var("J")

    # Get process parameters
    t = get_var("t") * (1 / Omega_ci) * 1000
    Ne = get_var("Ne") * Nnorm
    Nn = get_var("Nd") * Nnorm
    Te = get_var("Te") * Tnorm
    Tn = get_var("Td") * Tnorm

    res = {}
    for param in ["ddtPe", "ddtPi", "ddtPn", "ddtNe", "ddtNn", "ddtNVi", "ddtNVd"]:
        try:
            res[param] = get_var(param)
        except KeyError:
            res[param] = np.zeros_like(Ne)

    dv = dx * dz * J

    # Get solver parameters
    wtime = get_var("wtime")
    nliters = get_var("cvode_nliters")
    nniters = get_var("cvode_nniters")
    nfails = get_var("cvode_num_fails")
    lorder = get_var("cvode_last_order")

    print("..data", end="")

    # Selecting a point in the 1D simulation
    midpoint = len(Ne[0]) // 2

    # First row of plots
    Ne_mid = Ne[:, midpoint]
    Ne_max = np.max(Ne, axis=1)
    Nn_max = np.max(Nn, axis=1)
    Tn_max = np.max(Tn, axis=1)
    Te_max = np.max(Te, axis=1)

    # Second row of plots
    stime = np.diff(t, prepend=t[0] * 0.99)
    ms_per_24hrs = stime / (wtime / (60 * 60 * 24))  # ms simulated per 24 hours
    lratio = np.diff(nliters, prepend=nliters[1] * 0.99) / \
        np.diff(nniters, prepend=nniters[1] * 0.99)  # Ratio of linear to nonlinear iterations
    fails = np.diff(nfails, prepend=nfails[1] * 0.99)
    fails[0] = fails[1]
    lorder[0] = lorder[1]
    ms_per_24hrs[0] = ms_per_24hrs[1]

    # ddt calculations
    for param in res:
        res[param] = (res[param] * dv) / np.sum(dv)   # Volume weighted
        res[param] = np.sqrt(np.mean(res[param]**2, axis=1))  # RMS
        res[param] = np.convolve(res[param], np.ones(1), "same")  # Moving average with window of 1

    print("..calculations", end="")

    # Plotting
    if plot is True or save is True:
        scale = 1.2
        figsize = (8 * scale, 4 * scale)
        dpi = 150 / scale
        fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi)

        fig.subplots_adjust(hspace=0.4, top=0.85)
        fig.suptitle(casename, y=1.02)

        lw = 2
        axes[0, 0].plot(t, Ne_mid, c="darkorange", lw=lw)
        axes[0, 0].set_title("$N_{e}^{mid}$")

        if neutrals is True:
            axes[0, 3].plot(t, Tn_max, c="limegreen", lw=lw)
            axes[0, 3].set_title("$T_{n}^{max}$")

            axes[0, 1].plot(t, Tn_max, c="darkorchid", lw=lw)
            axes[0, 1].set_title("$T_{n}^{max}$")

            axes[0, 2].plot(t, Nn_max, c="deeppink", lw=lw)
            axes[0, 2].set_title("$N_{n}^{max}$")

        else:
            axes[0, 1].plot(t, Te_max, c="limegreen", lw=lw)
            axes[0, 1].set_title("$T_{e}^{max}$")

            axes[0, 2].plot(t, Ne_max, c="deeppink", lw=lw)
            axes[0, 2].set_title("$N_{e}^{max}$")

            axes[0, 3].plot(t, Te_max, c="darkorchid", lw=lw)
            axes[0, 3].set_title("$T_{e}^{max}$")

        axes[1, 0].plot(t, ms_per_24hrs, c="k", lw=lw)
        axes[1, 0].set_title("ms $t_{sim}$ / 24hr $t_{wall}$")

        axes[1, 1].plot(t, lratio, c="k", lw=lw)
        axes[1, 1].set_title("linear/nonlinear")

        axes[1, 2].plot(t, np.clip(fails, 0, np.max(fails)), c="k", lw=lw)
        axes[1, 2].set_title("nfails")
        axes[1, 2].set_ylim(0, None)

        axes[1, 3].plot(t, lorder, c="k", lw=lw)
        axes[1, 3].set_title("order")

        for i in [0, 1]:
            for ax in axes[i, :]:
                ax.grid(c="k", alpha=0.15)
                ax.xaxis.set_major_locator(
                    mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=5))
                ax.yaxis.set_major_locator(
                    mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=5))
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)

        fig.tight_layout()

        print("..figures", end="")

        if plot is True:
            plt.show()

        if save:
            fig.savefig(f"mon_{casename}.png", bbox_inches="tight",
                        pad_inches=0.2)
            print("..saved figures", end="")

    # Print table
    if table is True:
        def pad(x, l):
            num_spaces = l - len(x)
            return str(x) + " " * num_spaces

        def pad_minus(x):
            x = f"{x:.3e}"
            pad = "" if x[0] == "-" else " "
            return pad + x

        Nmidrate = np.gradient(Ne_mid, t) / Ne_mid

        print("..table\n\n")
        print("it |  t[ms] |    Nmid      Nmidrate |   Netarg  |  Nntarg  | Tnsol | w/s time | l/n |  nfails  | order |")

        for i, time in enumerate(t):
            Tnprint = pad(f"{Tn_max[i]:.1f}", 5)
            s1 = f"{pad(str(i), 2)} | {time:.2f} | {Ne_mid[i]:.3e}  {pad_minus(Nmidrate[i])} | {Ne_max[i]:.3e}"
            s2 = f" | {Nn_max[i]:.3e} | {Tnprint} | {ms_per_24hrs[i]:.2e} | {lratio[i]:.1f} | {nfails[i]:.2e} |   {lorder[i]:.0f}   |"

            print(s1 + s2)

    tend = tm.time()
    print(f"Executed in {tend-tstart:.1f} seconds")


#------------------------------------------------------------
# PARSER
#------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Case monitor")
    parser.add_argument("path", type=str, help="Path to case")
    parser.add_argument("-p", action="store_true", help="Plot?")
    parser.add_argument("-t", action="store_true", help="Table?")
    parser.add_argument("-s", action="store_true", help="Save figure?")
    parser.add_argument(
        "-neutrals", action="store_true", help="Neutral focused plot?")

    args = parser.parse_args()
    cmonitor(args.path, plot=args.p, table=args.t, save=args.s,
             neutrals=args.neutrals)
