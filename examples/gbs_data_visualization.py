import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from thewalrus.quantum import photon_number_covmat, photon_number_mean_vector

import strawberryfields as sf
from strawberryfields.ops import BSgate, LossChannel, MeasureFock, Rgate, Sgate
from strawberryfields.tdm import borealis_gbs, get_mode_indices
from strawberryfields.utils import gbs_runtime, gbs_sample_runtime

# convert different units of time to log10 years, needed for contour labels in
# the simulation-time plot
day = np.log10(1 / 365)
ten_days = np.log10(10 / 365)
hour = np.log10(1 / (365 * 24))
minute = np.log10(1 / (365 * 24 * 60))
second = np.log10(1 / (365 * 24 * 60 * 60))

# fontsizes used in the plots
fs_axlabel = 22
fs_text = 20
fs_ticklabel = 21
fs_legend = 20


def plot_photon_number_moments(mean_n, cov_n):
    """Plots first and second moment of the photon-number distribution

    Args:
        mean_n (array[float]): mean photon number per mode
        cov_n (array[int]): photon-number covariance matrix

    Returns:
        tuple: a pyplot ``figure`` and ``axis`` object
    """

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    ax[0].bar(range(len(mean_n)), mean_n, width=0.75, align="center")
    ax[0].set_title(
        rf"<$n$> = {np.mean(mean_n):.3f}, <$N$> = {np.sum(mean_n):.3f}",
        fontsize=fs_axlabel,
    )
    ax[0].set_xlabel("pulse index", fontsize=fs_axlabel)
    ax[0].set_ylabel("mean photon number", fontsize=fs_axlabel)
    ax[0].grid()
    ax[0].tick_params(axis="x", labelsize=fs_ticklabel)
    ax[0].tick_params(axis="y", labelsize=fs_ticklabel)

    ax[1].imshow(
        cov_n,
        norm=matplotlib.colors.SymLogNorm(linthresh=10e-6, linscale=1e-4, vmin=0, vmax=2),
        cmap="rainbow",
    )
    ax[1].set_title(r"Cov($n{_i}$, $n{_j}$)", fontsize=fs_axlabel)
    ax[1].tick_params(axis="x", labelsize=fs_ticklabel)
    ax[1].tick_params(axis="y", labelsize=fs_ticklabel)

    return fig, ax


def plot_photon_number_moment_comparison(mean_n_exp, mean_n_sim, cov_n_exp, cov_n_sim):
    """Compare in scatter plots the first and second moments of the photon-number distribution
    resulting from experiment and simulation.

    Args:
        mean_n_exp (array): experimental mean photon number per mode
        mean_n_sim (array): simulated mean photon number per mode
        cov_n_exp (array): experimental photon-number covariance matrix
        cov_n_sim (array): simulated photon-number covariance matrix

    Returns:
        tuple: a pyplot ``figure`` and ``axis`` object
    """

    cov_n_exp2 = np.copy(cov_n_exp)
    cov_n_sim2 = np.copy(cov_n_sim)

    # remove the diagonal elements (corresponding to the single-mode variance)
    # which would otherwise be dominant
    cov_n_exp2 -= np.diag(np.diag(cov_n_exp2))
    cov_n_sim2 -= np.diag(np.diag(cov_n_sim2))

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    min_ = np.min([mean_n_sim, mean_n_exp])
    max_ = np.max([mean_n_sim, mean_n_exp])
    ax[0].scatter(mean_n_sim, mean_n_exp, s=4, alpha=0.50)
    ax[0].plot([min_, max_], [min_, max_], "k--")
    ax[0].set_title("1st moment", fontsize=fs_axlabel)
    ax[0].set_xlabel("simulation", fontsize=fs_axlabel)
    ax[0].set_ylabel("experiment", fontsize=fs_axlabel)
    ax[0].set_xlim([min_, max_])
    ax[0].set_ylim([min_, max_])
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].tick_params(axis="x", labelsize=fs_ticklabel)
    ax[0].tick_params(axis="y", labelsize=fs_ticklabel)
    ax[0].grid()

    min_ = np.min([cov_n_sim2, cov_n_exp2])
    max_ = np.max([cov_n_sim2, cov_n_exp2])
    ax[1].scatter(cov_n_sim2, cov_n_exp2, s=4, alpha=0.50)
    ax[1].plot([min_, max_], [min_, max_], "k--")
    ax[1].set_title("2nd moment", fontsize=fs_axlabel)
    ax[1].set_xlabel("simulation", fontsize=fs_axlabel)
    ax[1].set_ylabel("experiment", fontsize=fs_axlabel)
    ax[1].set_xlim([min_, max_])
    ax[1].set_ylim([min_, max_])
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].tick_params(axis="x", labelsize=fs_ticklabel)
    ax[1].tick_params(axis="y", labelsize=fs_ticklabel)
    ax[1].grid()

    return fig, ax


def log_years_to_string(t):
    """Returns a string representation of the input time ``t``

    Args:
        t (float): time in units of log10 years (e.g. ``t==2``
            corresponds to 100 years)

    Returns:
        str: a nice exponential string representation of the time ``t``
    """

    if t == 1:
        return "10 years"
    if t == 0:
        return "1 year"
    if t >= 0:
        s = "{{{}}}".format(int(t))
        return rf"$10^{s}$ years"

    else:
        if t == ten_days:
            return "10 days"
        if t == day:
            return "1 day"
        if t == hour:
            return "1 hour"
        if t == minute:
            return "1 minute"
        if t == second:
            return "1 second"


def seconds_to_string(t):
    """Returns a string representation of the input time ``t``

    Args:
        t (float): time in seconds

    Returns:
        str: a nice exponential string representation of the time ``t``
    """
    t_years = t / 365 / 24 / 3600
    exponent = int(np.floor(np.log10(t_years)))
    scalar = t_years / 10**exponent
    exponent_string = "{{{}}}".format(int(exponent))
    return rf"{round(scalar)} x $10^{exponent_string}$"


def plot_simulation_time(samples):
    """Plots the simulation time of an input array of GBS samples

    Args:
        samples (array): a photon-number array of shape
            ``(shots, 1, temporal_modes)``

    Returns:
        tuple: a pyplot ``figure`` and ``axis`` object
    """

    # reshape the samples to remove the unnecessary ``spatial_modes`` dimension
    shots, spatial_modes, temporal_modes = samples.shape
    samples = samples.reshape((shots, temporal_modes))

    # compute runtime, N_c and G of each shot
    runtimes = np.zeros(shots)
    N_cs = np.zeros(shots, dtype=int)
    Gs = np.zeros(shots)
    for i, sample in enumerate(samples):
        runtimes[i], N_cs[i], Gs[i] = gbs_sample_runtime(sample, return_ncg=True)

    # the sample with highest runtime to be highlated below
    max_sample = {
        "r": np.max(runtimes),
        "N_c": N_cs[np.argmax(runtimes)],
        "G": Gs[np.argmax(runtimes)],
    }

    # create the x- and y-axis of the 2D histogram
    x_min = np.min(N_cs)
    x_max = np.max(N_cs)
    y_min = np.min(Gs)
    y_max = np.max(Gs)
    x = np.arange(x_min, x_max + 1)
    y = np.linspace(y_min, y_max, 50)

    # compute a runtime for each pixel in the 2D histogram (needed for the
    # contour labels)
    N_c_mesh, G_mesh = np.meshgrid(x, y)
    times = gbs_runtime(N_c_mesh, G_mesh, temporal_modes)
    log_times = np.log10(times / (365 * 24 * 60 * 60))

    # create a 2D histogram with the occurrence of each N_c and G
    x_bins = len(x)
    y_bins = len(y)
    hist, hist_xs, hist_ys = np.histogram2d(
        N_cs,
        Gs,
        bins=(x_bins, y_bins),
        range=((x_min, x_max), (y_min, y_max)),
    )

    fig, ax = plt.subplots(figsize=(18, 10))

    # plot the 2D histogram with the occurrence of each N_c and G
    im = ax.pcolormesh(
        hist_xs,
        hist_ys,
        np.log10(hist.T + 1e-15),
        cmap="Reds",
        vmin=0,
        vmax=np.log10(hist + 1e-15).max(),
        linewidth=0,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"# of samples [log$_{10}$]", fontsize=fs_legend)

    # highlight the sample with highest runtime
    ax.scatter(
        max_sample["N_c"],
        max_sample["G"],
        marker="*",
        color="red",
        s=300,
        label="brightest sample",
    )

    # contour labels to indicate the runtime throughout the histogram
    cs = ax.contour(
        x,
        y,
        log_times,
        levels=[minute, hour, day, ten_days] + [0, 1, 3, 6, 9, 12, 15, 18],
        colors="black",
    )
    ax.clabel(cs, cs.levels, fmt=log_years_to_string, fontsize=fs_text)

    ax.set_title(
        f"simulation runtime [years]:\n"
        f"median: {seconds_to_string(np.median(runtimes))}, average: {seconds_to_string(np.mean(runtimes))}, brightest: {seconds_to_string(np.max(runtimes))}, total: {seconds_to_string(np.sum(runtimes))}",
        fontsize=fs_axlabel,
    )
    ax.set_xlabel("$N_c$", fontsize=fs_axlabel)
    ax.set_ylabel("$G$", fontsize=fs_axlabel)
    ax.tick_params(axis="x", labelsize=fs_ticklabel)
    ax.tick_params(axis="y", labelsize=fs_ticklabel)
    ax.legend(fontsize=fs_legend)

    return fig, ax


if __name__ == "__main__":

    # connect to the remote engine and obtain a ``device`` object
    eng = sf.RemoteEngine("borealis")
    device = eng.device

    # create a list of list of gate arguments for a GBS instance
    gate_args_list = borealis_gbs(device, modes=288, squeezing="high")

    # create a Strawberry Fields program
    delays = [1, 6, 36]
    vac_modes = sum(delays)
    n, N = get_mode_indices(delays)
    prog = sf.TDMProgram(N)
    with prog.context(*gate_args_list) as (p, q):
        Sgate(p[0]) | q[n[0]]
        for i in range(len(delays)):
            Rgate(p[2 * i + 1]) | q[n[i]]
            BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
        MeasureFock() | q[0]

    # define number of shots and submit job to hardware
    shots = 250_000
    results = eng.run(prog, shots=shots, crop=True)

    # the GBS samples
    samples = results.samples

    # plot the estimated simulation times of the experimental samples
    plot_simulation_time(samples)

    # obtain first and second moment of the photon-number distribution: mean photon
    # number and photon-number covariance
    mean_n = np.mean(samples, axis=(0, 1))
    cov_n = np.cov(samples[:, 0, :].T)

    # plot first and second moment of the experimental samples
    plot_photon_number_moments(mean_n, cov_n)

    # the common efficiency -- ``1 - eta_glob`` would correspond to the minimum loss
    # that each mode suffers
    eta_glob = device.certificate["common_efficiency"]

    # a list with the efficiencies of the three loops
    etas_loop = device.certificate["loop_efficiencies"]

    # a list with the relative efficiencies of the 16 detector channels
    etas_ch_rel = device.certificate["relative_channel_efficiencies"]

    # the detector-channel efficiencies are cyclic, so mode ``i`` sees the same
    # efficiency as mode ``i+1``
    prog_length = len(gate_args_list[0])
    reps = int(np.ceil(prog_length / 16))
    etas_ch_rel = np.tile(etas_ch_rel, reps)[:prog_length]

    # create a Strawberry Fields program for a local simulation
    prog_sim = sf.TDMProgram(N)
    with prog_sim.context(*gate_args_list, etas_ch_rel) as (p, q):
        Sgate(p[0]) | q[n[0]]
        LossChannel(eta_glob) | q[n[0]]
        for i in range(len(delays)):
            Rgate(p[2 * i + 1]) | q[n[i]]
            BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
            LossChannel(etas_loop[i]) | q[n[i]]
        LossChannel(p[7]) | q[0]
        MeasureFock() | q[0]
    prog_sim.space_unroll()

    # create Gaussian engine and submit program to it
    eng_sim = sf.Engine(backend="gaussian")
    results_sim = eng_sim.run(prog_sim, shots=None, crop=True)

    # obtain quadrature covariance matrix
    cov = results_sim.state.cov()

    # obtain first and second moment
    mu = np.zeros(len(cov))
    mean_n_sim = photon_number_mean_vector(mu, cov)
    cov_n_sim = photon_number_covmat(mu, cov)

    # plot first and second moment of the simulated data
    plot_photon_number_moments(mean_n_sim, cov_n_sim)

    # plot statistical comparison between data and simulation
    plot_photon_number_moment_comparison(mean_n, mean_n_sim, cov_n, cov_n_sim)

    # show all plots
    plt.show()
