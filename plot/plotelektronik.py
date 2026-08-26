import numpy as np
#plt.style.use('../matplotlib/sci.mplstyle')
import matplotlib.pyplot as plt

# ==========================================================
# INPUT FILE
# ==========================================================

band_file = "mos2.band.gnu"
dos_file  = "mos2.dos"

# Fermi energy (eV)
highest = 1.2398
Lowest = 2.8455
Ef = (highest +Lowest)/2

# Rentang energi yang ditampilkan
Emin = -4
Emax =  4

# Batas maksimum DOS
DOS_max = 7


# ==========================================================
# TITIK SIMETRI
# ==========================================================

# Sesuaikan dengan jumlah titik-k pada band.gnu
high_symmetry_index = [0, 40, 80, -1]

high_symmetry_label = [
    r"$\Gamma$",
    "M",
    "K",
    r"$\Gamma$"
]


# ==========================================================
# MEMBACA FILE BAND.GNU
# ==========================================================

bands = []
current_band = []

with open(band_file, "r") as f:

    for line in f:

        line = line.strip()

        # Baris kosong = pita berikutnya
        if line == "":

            if current_band:
                bands.append(
                    np.array(current_band)
                )
                current_band = []

            continue

        # Abaikan komentar
        if line.startswith("#"):
            continue

        data = line.split()

        if len(data) >= 2:

            k = float(data[0])
            energy = float(data[1])

            current_band.append(
                [k, energy]
            )


# Tambahkan pita terakhir
if current_band:

    bands.append(
        np.array(current_band)
    )


# Koordinat k
kdist = bands[0][:, 0]


# ==========================================================
# MEMBACA DOS
# ==========================================================

dos_data = np.loadtxt(
    dos_file,
    comments="#"
)

energy_dos = dos_data[:, 0]
dos_value  = dos_data[:, 1]


# ==========================================================
# SHIFT TERHADAP FERMI ENERGY
# ==========================================================

# Band
bands_shifted = []

for band in bands:

    energy = band[:, 1] - Ef

    bands_shifted.append(
        energy
    )


# DOS
energy_dos = energy_dos - Ef


# ==========================================================
# POSISI TITIK SIMETRI
# ==========================================================

symmetry_positions = []

for index in high_symmetry_index:

    if index == -1:

        index = len(kdist) - 1

    symmetry_positions.append(
        kdist[index]
    )


# ==========================================================
# MEMBUAT FIGURE
# ==========================================================

fig, (ax_band, ax_dos) = plt.subplots(
    1,
    2,

    figsize=(7, 5),

    # Band lebih lebar daripada DOS
    gridspec_kw={
        "width_ratios": [3.5, 1]
    },

    # Sumbu Y sama
    sharey=True
)


# ==========================================================
# BAND STRUCTURE
# ==========================================================

for energy in bands_shifted:

    ax_band.plot(
        kdist,
        energy,
        linewidth=1.3,
        color="purple"
    )


# ----------------------------------------------------------
# Garis vertikal titik simetri
# ----------------------------------------------------------

for x in symmetry_positions:

    ax_band.axvline(
        x,
        color="gray",
        linewidth=0.8
    )


# ----------------------------------------------------------
# Fermi level
# ----------------------------------------------------------

ax_band.axhline(
    0,
    color="gray",
    linestyle="--",
    linewidth=0.8
)


# ----------------------------------------------------------
# Batas sumbu
# ----------------------------------------------------------

ax_band.set_xlim(
    kdist[0],
    kdist[-1]
)

ax_band.set_ylim(
    Emin,
    Emax
)


# ----------------------------------------------------------
# Label titik simetri
# ----------------------------------------------------------

ax_band.set_xticks(
    symmetry_positions
)

ax_band.set_xticklabels(
    high_symmetry_label,
    fontsize=11
)


# ----------------------------------------------------------
# Label Y
# ----------------------------------------------------------

ax_band.set_ylabel(
    r"$E-E_F$ (eV)",
    fontsize=12
)


# ==========================================================
# DOS
# ==========================================================

ax_dos.plot(
    dos_value,
    energy_dos,
    color="purple",
    linewidth=1.3
)


# ----------------------------------------------------------
# Isi / shading DOS
# ----------------------------------------------------------

ax_dos.fill_betweenx(
    energy_dos,
    0,
    dos_value,
    color="purple",
    alpha=0.20
)


# ----------------------------------------------------------
# Fermi level
# ----------------------------------------------------------

ax_dos.axhline(
    0,
    color="gray",
    linestyle="--",
    linewidth=0.8
)


# ----------------------------------------------------------
# Batas DOS
# ----------------------------------------------------------

ax_dos.set_xlim(
    0,
    DOS_max
)

ax_dos.set_ylim(
    Emin,
    Emax
)


# ----------------------------------------------------------
# Label DOS
# ----------------------------------------------------------

ax_dos.set_xlabel(
    "DOS",
    fontsize=11
)


# Hilangkan label Y di panel DOS
ax_dos.tick_params(
    axis="y",
    labelleft=False
)


# ==========================================================
# TAMPILAN
# ==========================================================

ax_band.set_title(
    "Band Structure",
    fontsize=11
)

ax_dos.set_title(
    "DOS",
    fontsize=11
)


# Hilangkan jarak antara band dan DOS
plt.subplots_adjust(
    wspace=0.05
)


# ==========================================================
# SIMPAN
# ==========================================================

plt.savefig(
    "band_DOS.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "band_DOS.pdf",
    bbox_inches="tight"
)


plt.show()