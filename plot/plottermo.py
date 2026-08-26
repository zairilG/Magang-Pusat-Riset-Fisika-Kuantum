#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CONDTENS_FILE = "mos2.condtens"

# GANTI dengan energi referensi DFT yang sama dengan plot band dan DOS.
highest = 1.2398
lowest = 2.8456
FERMI_ENERGY_EV = (highest + lowest)/2

TEMPERATURES_K = [300.0, 500.0, 700.0, 900.0]
MU_WINDOW_EV = (-1.5, 1.5)

# None berarti plot sigma/tau, kappa_e/tau, dan PF/tau.
# Masukkan nilai numerik hanya jika tau diketahui atau diasumsikan.
TAU_SECONDS = None

# Normalisasi monolayer.
APPLY_2D_RESCALE = False
LZ_ANGSTROM = 20.0
D_EFFECTIVE_ANGSTROM = 6.15

OUTPUT_PNG = Path("figures/mos2_thermoelectric_vs_mu.png")
OUTPUT_PDF = Path("figures/mos2_thermoelectric_vs_mu.pdf")

RY_TO_EV = 13.605693122994


def read_condtens(filename):
    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 30:
        raise RuntimeError(
            f"{filename} hanya memiliki {data.shape[1]} kolom."
        )

    return data


def in_plane_average(data, col_xx, col_yy):
    return 0.5 * (data[:, col_xx] + data[:, col_yy])


def main():
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    data = read_condtens(CONDTENS_FILE)

    mu_abs_ev = data[:, 0] * RY_TO_EV
    mu_rel_ev = mu_abs_ev - FERMI_ENERGY_EV
    temperature = data[:, 1]

    sigma_over_tau = in_plane_average(data, 3, 7)
    seebeck_v_per_k = in_plane_average(data, 12, 16)
    kappa_over_tau = in_plane_average(data, 21, 25)

    if APPLY_2D_RESCALE:
        factor = LZ_ANGSTROM / D_EFFECTIVE_ANGSTROM
        sigma_over_tau = sigma_over_tau * factor
        kappa_over_tau = kappa_over_tau * factor

    pf_over_tau = seebeck_v_per_k**2 * sigma_over_tau
    seebeck_uv_per_k = seebeck_v_per_k * 1.0e6

    if TAU_SECONDS is None:
        sigma_plot = sigma_over_tau
        kappa_plot = kappa_over_tau
        pf_plot = pf_over_tau

        sigma_ylabel = r"$\sigma_{\parallel}/\tau$ [$(\Omega\,m\,s)^{-1}$]"
        kappa_ylabel = r"$\kappa_{e,\parallel}/\tau$ [W m$^{-1}$ K$^{-1}$ s$^{-1}$]"
        pf_ylabel = r"$PF_{\parallel}/\tau$ [W m$^{-1}$ K$^{-2}$ s$^{-1}$]"
    else:
        sigma_plot = sigma_over_tau * TAU_SECONDS
        kappa_plot = kappa_over_tau * TAU_SECONDS
        pf_plot = pf_over_tau * TAU_SECONDS

        sigma_ylabel = r"$\sigma_{\parallel}$ [S m$^{-1}$]"
        kappa_ylabel = r"$\kappa_{e,\parallel}$ [W m$^{-1}$ K$^{-1}$]"
        pf_ylabel = r"$PF_{\parallel}$ [W m$^{-1}$ K$^{-2}$]"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 8.0),
        sharex=True,
    )

    ax_s, ax_sigma, ax_kappa, ax_pf = axes.ravel()

    for target_T in TEMPERATURES_K:
        mask = np.isclose(
            temperature,
            target_T,
            rtol=0.0,
            atol=1.0e-8,
        )

        if not np.any(mask):
            print(f"T = {target_T:g} K tidak ditemukan.")
            continue

        x = mu_rel_ev[mask]
        s = seebeck_uv_per_k[mask]
        sig = sigma_plot[mask]
        kap = kappa_plot[mask]
        pf = pf_plot[mask]

        order = np.argsort(x)

        x = x[order]
        s = s[order]
        sig = sig[order]
        kap = kap[order]
        pf = pf[order]

        window = (
            (x >= MU_WINDOW_EV[0])
            &
            (x <= MU_WINDOW_EV[1])
        )

        x = x[window]
        s = s[window]
        sig = sig[window]
        kap = kap[window]
        pf = pf[window]

        label = f"{target_T:g} K"

        ax_s.plot(x, s, label=label)
        ax_sigma.plot(x, sig, label=label)
        ax_kappa.plot(x, kap, label=label)
        ax_pf.plot(x, pf, label=label)

    for ax in axes.ravel():
        ax.axvline(0.0, linewidth=0.9, linestyle="--")
        ax.set_xlim(*MU_WINDOW_EV)

    ax_s.axhline(0.0, linewidth=0.8, linestyle=":")
    ax_s.set_ylabel(r"$S_{\parallel}$ [$\mu$V/K]")
    ax_s.set_title("Koefisien Seebeck")

    ax_sigma.set_ylabel(sigma_ylabel)
    ax_sigma.set_title("Konduktivitas listrik")

    ax_kappa.set_ylabel(kappa_ylabel)
    ax_kappa.set_title("Konduktivitas termal elektronik")
    ax_kappa.set_xlabel(r"$\mu-E_F$ (eV)")

    ax_pf.set_ylabel(pf_ylabel)
    ax_pf.set_title("Power factor")
    ax_pf.set_xlabel(r"$\mu-E_F$ (eV)")

    for ax in axes.ravel():
        ax.legend(frameon=False)

    fig.suptitle(
        "Monolayer MoS$_2$: sifat termoelektrik terhadap potensial kimia"
    )
    fig.tight_layout()

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()