#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import boutdata
from boututils.options import BOUTOptions
from boutdata.collect import create_cache
import argparse
import os
import time as tm


def cmonitor_1d(path, save=False, plot=False, table=True, neutrals=False):
    """
    Produce convergence report for 1D Hermes-3 simulation.

    Inputs
    -----
    path: str, path to case directory
    save: bool, save figure, saved by case name
    plot: bool, show plot
    table: bool, show table
    neutrals: bool, handle neutral parameters
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
            xguards=True,  # Always with guards to minimise mistakes
            strict=True,  # To prevent reading wrong variable by accident
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
    for param in ["ddtPe", "ddtNe", "ddtNn"]:
        try:
            res[param] = get_var(param)
        except:
            res[param] = np.zeros_like(Ne)

    dv = dx * dz * J

    # Get solver parameters
    wtime = get_var("wtime")
    nliters = get_var("cvode_nliters")
    nniters = get_var("cvode_nniters")
    nfails = get_var("cvode_num_fails")
    lorder = get_var("cvode_last_order")

    print("..data", end="")

    # Calculate locations (in 1D, these are simpler)
    Ne_target = Ne[:, -1]  # Ne at the farthest point in 1D
    Tn_target = Tn[:, -1]
    Te_target = Te[:, -1]

    # Prepare for plotting
    if plot is True or save is True:
        scale = 1.2
        figsize = (8 * scale, 4 * scale)
        dpi = 150 / scale
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(hspace=0.4, top=0.85)
        fig.suptitle(casename, y=1.02)

        lw = 2
        axes[0, 0].plot(t, Ne_target, c="darkorange", lw=lw)
        axes[0, 0].set_title("$N_{e}^{target}$")

        if neutrals is True:
            axes[0, 1].plot(t, Tn_target, c="limegreen", lw=lw)
            axes[0, 1].set_title("$T_{n}^{target}$")
        else:
            axes[0, 1].plot(t, Te_target, c="limegreen", lw=lw)
            axes[0, 1].set_title("$T_{e}^{target}$")

        # Solver performance plots
        axes[1, 0].plot(t, wtime, c="k", lw=lw)
        axes[1, 0].set_title("Wall Time (s)")
        axes[1, 1].plot(t, lorder, c="k", lw=lw)
        axes[1, 1].set_title("Order")

        for i in [0, 1]:
            for ax in axes[i, :]:
                ax.grid(c="k", alpha=0.15)
                ax.xaxis.set_major_locator(plt.MaxNLocator(min_n_ticks=3, nbins=5))
                ax.yaxis.set_major_locator(plt.MaxNLocator(min_n_ticks=3, nbins=5))
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

        fig.tight_layout()
        print("..figures", end="")

        if plot is True:
            plt.show()

        if save:
            fig.savefig(f"mon_{casename}.png", bbox_inches="tight", pad_inches=0.2)
            print("..saved figures", end="")

    # Print table
    if table is True:
        print("..table\n\n")
        print("it | t[ms] | Ne_target | Te_target | wtime | order")
        for i, time in enumerate(t):
            print(f"{i:2d} | {time:.2f} | {Ne_target[i]:.3e} | "
                  f"{Te_target[i]:.3e} | {wtime[i]:.2e} | {lorder[i]:.0f}")

    tend = tm.time()
    print(f"Executed in {tend - tstart:.1f} seconds")


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Case monitor")
    parser.add_argument("path", type=str, help="Path to case")
    parser.add_argument("-p", action="store_true", help="Plot?")
    parser.add_argument("-t", action="store_true", help="Table?")
    parser.add_argument("-s", action="store_true", help="Save figure?")
    parser.add_argument("-neutrals", action="store_true", help="Neutral-focused plot?")

    # Extract arguments and call function
    args = parser.parse_args()
    cmonitor_1d(args.path, plot=args.p, table=args.t, save=args.s, neutrals=args.neutrals)
