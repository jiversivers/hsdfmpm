import struct
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from numpy.polynomial.polynomial import polyfit
from pydantic import BaseModel, model_validator, field_validator


def read_chars(f, count):
    return f.read(count).decode("latin-1").strip("\x00")


def read_fmt(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def continue_parsing_measurement_info(f, mi):
    # FCS Info
    fcs = {}
    fcs["chan"] = read_fmt(f, "<H")
    fcs["fcs_decay_calc"] = read_fmt(f, "<H")
    fcs["mt_resol"] = read_fmt(f, "<I")
    fcs["cortime"] = read_fmt(f, "<f")
    fcs["calc_photons"] = read_fmt(f, "<I")
    fcs["fcs_points"] = read_fmt(f, "<i")
    fcs["end_time"] = read_fmt(f, "<f")
    fcs["overruns"] = read_fmt(f, "<H")
    fcs["fcs_type"] = read_fmt(f, "<H")
    fcs["cross_chan"] = read_fmt(f, "<H")
    fcs["mod"] = read_fmt(f, "<H")
    fcs["cross_mod"] = read_fmt(f, "<H")
    fcs["cross_mt_resol"] = read_fmt(f, "<I")
    mi["FCSInfo"] = fcs

    mi["image_x"] = read_fmt(f, "<i")
    mi["image_y"] = read_fmt(f, "<i")
    mi["image_rx"] = read_fmt(f, "<i")
    mi["image_ry"] = read_fmt(f, "<i")
    mi["xy_gain"] = read_fmt(f, "<h")
    mi["dig_flags"] = read_fmt(f, "<h")
    mi["adc_de"] = read_fmt(f, "<h")
    mi["det_type"] = read_fmt(f, "<h")
    mi["x_axis"] = read_fmt(f, "<h")

    # HIST Info
    hist = {}
    hist["fida_time"] = read_fmt(f, "<f")
    hist["filda_time"] = read_fmt(f, "<f")
    hist["fida_points"] = read_fmt(f, "<i")
    hist["filda_points"] = read_fmt(f, "<i")
    hist["mcs_time"] = read_fmt(f, "<f")
    hist["mcs_points"] = read_fmt(f, "<i")
    hist["cross_calc_phot"] = read_fmt(f, "<I")
    hist["mcsta_points"] = read_fmt(f, "<H")
    hist["mcsta_flags"] = read_fmt(f, "<H")
    hist["mcsta_tpp"] = read_fmt(f, "<I")
    hist["calc_markers"] = read_fmt(f, "<I")
    hist["fcs_calc_phot"] = read_fmt(f, "<I")
    hist["reserved3"] = read_fmt(f, "<I")
    mi["measurementInfoHISTInfo"] = hist

    # HIST Info Extension
    hist_ext = {}
    hist_ext["first_frame_time"] = read_fmt(f, "<f")
    hist_ext["frame_time"] = read_fmt(f, "<f")
    hist_ext["line_time"] = read_fmt(f, "<f")
    hist_ext["pixel_time"] = read_fmt(f, "<f")
    hist_ext["scan_type"] = read_fmt(f, "<h")
    hist_ext["skip_2nd_line_clk"] = read_fmt(f, "<h")
    hist_ext["right_border"] = read_fmt(f, "<I")
    hist_ext["info"] = read_chars(f, 40)
    mi["measurementInfoHISTInfoExt"] = hist_ext

    mi["sync_delay"] = read_fmt(f, "<f")
    mi["sdel_ser_no"] = read_fmt(f, "<H")
    mi["sdel_input"] = read_fmt(f, "B")
    mi["mosaic_ctrl"] = read_fmt(f, "B")
    mi["mosaic_x"] = read_fmt(f, "B")
    mi["mosaic_y"] = read_fmt(f, "B")
    mi["frames_per_el"] = read_fmt(f, "<h")
    mi["chan_per_el"] = read_fmt(f, "<h")
    mi["mosaic_cycles_done"] = read_fmt(f, "<i")
    mi["mla_ser_no"] = read_fmt(f, "<H")
    mi["DCC_in_use"] = read_fmt(f, "B")
    mi["dcc_ser_no"] = read_chars(f, 12)
    mi["TiSaLas_status"] = read_fmt(f, "<H")
    mi["TiSaLas_wav"] = read_fmt(f, "<H")
    mi["AOM_status"] = read_fmt(f, "B")
    mi["AOM_power"] = read_fmt(f, "B")
    mi["ddg_ser_no"] = read_chars(f, 8)
    mi["prior_ser_no"] = read_fmt(f, "<i")
    mi["mosaic_x_hi"] = read_fmt(f, "B")
    mi["mosaic_y_hi"] = read_fmt(f, "B")
    mi["reserve"] = read_chars(f, 12)

    return mi


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    return obj


def open_sdt_file_full_with_metadata(filename):
    with open(filename, "rb") as f:
        out = {}

        out["headerInfo"] = {
            "revision": read_fmt(f, "<h"),
            "info_offset": read_fmt(f, "<l"),
            "info_length": read_fmt(f, "<h"),
            "setup_offs": read_fmt(f, "<l"),
            "setup_length": read_fmt(f, "<h"),
            "data_block_offset": read_fmt(f, "<l"),
            "no_of_data_blocks": read_fmt(f, "<h"),
            "data_block_length": read_fmt(f, "<L"),
            "meas_desc_block_offset": read_fmt(f, "<l"),
            "no_of_meas_desc_blocks": read_fmt(f, "<h"),
            "meas_desc_block_length": read_fmt(f, "<h"),
            "header_valid": read_fmt(f, "<H"),
            "reserved1": read_fmt(f, "<L"),
            "reserved2": read_fmt(f, "<H"),
            "chksum": read_fmt(f, "<H"),
        }

        out["fileInformation"] = f.read(out["headerInfo"]["info_length"]).decode(
            "latin-1"
        )
        out["setup"] = f.read(out["headerInfo"]["setup_length"]).decode("latin-1")

        # Measurement Info
        mi = {}
        mi["time"] = read_chars(f, 9)
        mi["date"] = read_chars(f, 11)
        mi["mod_ser_no"] = read_chars(f, 16)
        mi["measurementInfo_mode"] = read_fmt(f, "<h")
        mi["cfd_ll"] = read_fmt(f, "<f")
        mi["cfd_lh"] = read_fmt(f, "<f")
        mi["cfd_zc"] = read_fmt(f, "<f")
        mi["cfd_hf"] = read_fmt(f, "<f")
        mi["syn_zc"] = read_fmt(f, "<f")
        mi["syn_fd"] = read_fmt(f, "<h")
        mi["syn_hf"] = read_fmt(f, "<f")
        mi["tac_r"] = read_fmt(f, "<f")
        mi["tac_g"] = read_fmt(f, "<h")
        mi["tac_of"] = read_fmt(f, "<f")
        mi["tac_ll"] = read_fmt(f, "<f")
        mi["tac_lh"] = read_fmt(f, "<f")
        mi["adc_re"] = read_fmt(f, "<h")
        mi["eal_de"] = read_fmt(f, "<h")
        mi["ncx"] = read_fmt(f, "<h")
        mi["ncy"] = read_fmt(f, "<h")
        mi["page"] = read_fmt(f, "<H")
        mi["col_t"] = read_fmt(f, "<f")
        mi["rep_t"] = read_fmt(f, "<f")
        mi["stopt"] = read_fmt(f, "<h")
        mi["overfl"] = read_fmt(f, "B")
        mi["use_motor"] = read_fmt(f, "<h")
        mi["steps"] = read_fmt(f, "<H")
        mi["offset"] = read_fmt(f, "<f")
        mi["dither"] = read_fmt(f, "<h")
        mi["incr"] = read_fmt(f, "<h")
        mi["mem_bank"] = read_fmt(f, "<h")
        mi["mod_type"] = read_chars(f, 16)
        mi["syn_th"] = read_fmt(f, "<f")
        mi["dead_time_comp"] = read_fmt(f, "<h")
        mi["polarity_l"] = read_fmt(f, "<h")
        mi["polarity_f"] = read_fmt(f, "<h")
        mi["polarity_p"] = read_fmt(f, "<h")
        mi["linediv"] = read_fmt(f, "<h")
        mi["accumulate"] = read_fmt(f, "<h")
        mi["flbck_y"] = read_fmt(f, "<i")
        mi["flbck_x"] = read_fmt(f, "<i")
        mi["bord_u"] = read_fmt(f, "<i")
        mi["bord_l"] = read_fmt(f, "<i")
        mi["pix_time"] = read_fmt(f, "<f")
        mi["pix_clk"] = read_fmt(f, "<h")
        mi["trigger"] = read_fmt(f, "<h")
        mi["scan_x"] = read_fmt(f, "<i")
        mi["scan_y"] = read_fmt(f, "<i")
        mi["scan_rx"] = read_fmt(f, "<i")
        mi["scan_ry"] = read_fmt(f, "<i")
        mi["fifo_typ"] = read_fmt(f, "<h")
        mi["epx_div"] = read_fmt(f, "<i")
        mi["mod_type_code"] = read_fmt(f, "<H")
        mi["mod_fpga_ver"] = read_fmt(f, "<H")
        mi["overflow_corr_factor"] = read_fmt(f, "<f")
        mi["adc_zoom"] = read_fmt(f, "<i")
        mi["cycles"] = read_fmt(f, "<i")

        # StopInfo block
        stop = {
            "status": read_fmt(f, "<H"),
            "flags": read_fmt(f, "<H"),
            "stop_time": read_fmt(f, "<f"),
            "cur_step": read_fmt(f, "<i"),
            "cur_cycle": read_fmt(f, "<i"),
            "cur_page": read_fmt(f, "<i"),
            "min_sync_rate": read_fmt(f, "<f"),
            "min_cfd_rate": read_fmt(f, "<f"),
            "min_tac_rate": read_fmt(f, "<f"),
            "min_adc_rate": read_fmt(f, "<f"),
            "max_sync_rate": read_fmt(f, "<f"),
            "max_cfd_rate": read_fmt(f, "<f"),
            "max_tac_rate": read_fmt(f, "<f"),
            "max_adc_rate": read_fmt(f, "<f"),
            "reserved1": read_fmt(f, "<i"),
            "reserved2": read_fmt(f, "<f"),
        }
        mi["StopInfo"] = stop

        mi = continue_parsing_measurement_info(f, mi)
        out["measurementInfo"] = mi

        # Decay block
        out["decayInfo"] = {
            "data_offs_ext": read_fmt(f, "<B"),
            "next_block_offs_ext": read_fmt(f, "<B"),
            "data_offs": read_fmt(f, "<L"),
            "next_block_offs": read_fmt(f, "<L"),
            "block_type": read_fmt(f, "<H"),
            "meas_desc_block_no": read_fmt(f, "<h"),
            "lblock_no": read_fmt(f, "<L"),
            "block_length": read_fmt(f, "<L"),
        }

        # Decay data
        decay_data = np.fromfile(f, dtype=np.uint16)
        decay_data = decay_data.astype(np.float32)
        ny, nx, adc = mi["image_y"], mi["image_x"], mi["adc_re"]
        decay = decay_data.reshape(-1, ny, nx, adc)
        out["decay"] = decay
    return out


def open_sdt_file_with_json_metadata(filename):
    out = open_sdt_file_full_with_metadata(filename)
    json_metadata = make_json_serializable(
        {
            "headerInfo": out["headerInfo"],
            "fileInformation": out["fileInformation"],
            "setup": out["setup"],
            "measurementInfo": out["measurementInfo"],
            "decayInfo": out["decayInfo"],
        }
    )
    return out["decay"], json_metadata


def polar_from_cartesian(
        x: np.ndarray[float],
        y: np.ndarray[float]) -> tuple[np.ndarray[float], np.ndarray[float]]:
    z = x + 1j * y
    return np.angle(z), np.abs(z)


def cartesian_from_polar(theta: np.ndarray[float],
                         r: np.ndarray[float],
                         as_complex: bool = False) -> np.ndarray[float]:
    z = r * np.exp(1j * theta)
    if as_complex:
        return z
    return z.real, z.imag


def lifetime_from_cartesian(x: float, y: float, omega: float, harmonic: int = 1):
    return (1 / omega) * np.tan(np.arctan2(y, x))


def cartesian_from_lifetime(tau: float,
                            omega: float,
                            harmonic: int = 1,
                            as_complex: bool = False) -> np.ndarray[float]:
    phi, m = polar_from_lifetime(tau, omega, harmonic)
    return cartesian_from_polar(phi, m, as_complex)


def polar_from_lifetime(tau: float, omega: float, harmonic: int = 1):
    phi = np.arctan(harmonic * omega * tau)
    m = 1 / np.sqrt(1 + (harmonic * omega * tau) ** 2)
    return phi, m

def complex_phasor(g: np.ndarray[float], s: np.ndarray[float]) -> np.ndarray[complex]:
    return g + 1j * s


def plot_universal_circle(
    omega: float,
    harmonic: int = 1,
    tau_labels: Optional[list[float]] = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Axes, np.ndarray[float]]:
    # Get ax
    ax = ax if ax is not None else plt.gca()

    # Add Circle
    circle = Circle(
        (0.5, 0),
        radius=0.5,
        facecolor="none",
        edgecolor="black",
        label="Universal Circle",
    )
    ax.add_patch(circle)

    # Add labels
    if tau_labels is not None:
        xy_coords = np.zeros((len(tau_labels), 2), dtype=np.float64)
        for i, label in enumerate(tau_labels):
            x, y = cartesian_from_lifetime(label, harmonic * omega)
            ax.scatter(x, y, s=5, color="black", label="_nolegend_")
            xy_coords[i] = [x, y]
    else:
        xy_coords = np.array([], dtype=np.float64)
    return ax, xy_coords


# TODO: Incorporate into LifetimeImage
# TODO: Add density plot, universal circle, etc.
class PhasorPlot(BaseModel):
    g: np.ndarray
    s: np.ndarray
    line: Union[tuple[float, float], list[float, float], np.ndarray[float, float]] = (
        None
    )
    labelled_taus: Optional[dict[str, float]] = None
    frequency: float = 80e6
    harmonic: int = 1
    fig: Optional[plt.Figure] = None
    ax: Optional[plt.Axes] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @model_validator(mode="after")
    def start_plot(self):
        # Setup figure and axes
        f, a = plt.subplots()
        if self.ax is None:
            self.ax = a
        if self.fig is None:
            self.fig = f
        self.draw()
        return self

    def draw(self):
        # Scatter data
        self.ax.clear()
        self.scatter = self.ax.scatter(
            self.g.flatten(),
            self.s.flatten(),
            s=0.5,
            color="black",
            label="Phasor Cloud",
        )

        # Line data
        if self.line:
            m, b = self.line
            x_vals = np.linspace(0, 1, 100)
            y_vals = m * x_vals + b
            (self.line_plot,) = self.ax.plot(
                x_vals, y_vals, color="red", label="Fit line"
            )

        # Lifetime labels
        if self.labelled_taus:
            xs, ys = [], []
            for label, tau in self.labels.items():
                x, y = cartesian_from_lifetime(
                    tau, 2 * np.pi * self.frequency * self.harmonic
                )
                xs.append(x)
                ys.append(y)
                self.label_plot.append(
                    self.ax.annotate(label, x, y), xytext=(1.05 * x, 1.05 * y)
                )
            self.tau_plot = self.ax.scatter(
                xs, ys, color="blue", marker="x", label="_nolegend_"
            )

        # Ax settings
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 0.7)
        self.ax.set_xlabel("G")
        self.ax.set_ylabel("S")
        self.ax.set_aspect("equal")
        self.ax.legend()

    def toggle(self, **kwargs):
        for attr, option in kwargs.items():
            to_set = getattr(self, attr)
            to_set.set_visible(option)
        self.fig.canvas.draw_idle()

    def update_data(self, **kwargs):
        for attr, value in kwargs.items():
            if getattr(self, attr) is not None:
                setattr(self, attr, value)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def get_phasor_coordinates(
    decay: np.ndarray[float],
    bin_width: float = 10 / 256 / 1e9,
    frequency: float = 80e6,
    harmonic: int = 1,
    threshold: float = 0,
    as_complex: bool = False) -> np.ndarray:
    """Helper function to get the raw phasor coordinates from a decay curve given the imaging parameters provided."""
    T = decay.shape[-1]
    dt = bin_width
    w = 2 * np.pi * harmonic * frequency

    # Create an array of t (bin numbers) for each time frame of the decay
    t = np.expand_dims(np.arange(0.5, T, 1) * dt, axis=0)
    while t.ndim < decay.ndim:
        t = np.expand_dims(t, axis=0)

    # Calculate raw phasor coordinates
    g = np.sum(decay * np.cos(w * t), axis=-1)
    s = np.sum(decay * np.sin(w * t), axis=-1)
    photons = np.sum(decay, axis=-1)

    # Normalize to counts and zero pixels below threshold count
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(photons > threshold, g / photons, 0)
        s = np.where(photons > threshold, s / photons, 0)
    if as_complex:
        return complex_phasor(g, s), photons
    return g, s, photons


def fit_phasor(
        g: np.ndarray[float],
        s: np.ndarray[float],
        ratio_threshold=3) -> tuple[float, float]:
    vT, ratio, mu = phasor_svd(g, s)
    if ratio < ratio_threshold:
        return np.nan, np.nan
    return convert_vT_to_point_slope(vT, mu)


def convert_vT_to_point_slope(
        vT: np.ndarray[float, float],
        mu: np.ndarray[float, float]) -> tuple[float, float]:
    dx, dy = vT[0]
    m = dy / dx
    b = -m * mu[0] + mu[1]
    return b, m


def phasor_svd(
        g: np.ndarray[float],
        s: np.ndarray[float]) -> tuple[np.ndarray[float, float], float, np.ndarray[float, float]]:
    cloud = np.stack([g.flatten(), s.flatten()], axis=1)
    mu = cloud.mean(axis=0)
    cloud -= mu  # Center cloud
    _, s, vT = np.linalg.svd(cloud, full_matrices=False)
    return vT, s[0] / s[1], mu


def find_intersection_with_circle(
    b: float, m: float
) -> tuple[np.ndarray[float, float], np.ndarray[float, float]]:
    # Calculate intersection points of the line (quadratic eqn)
    x = (
        -(2 * m * b - 1)
        + np.array([1, -1]) * np.sqrt((2 * m * b - 1) ** 2 - 4 * (m**2 + 1) * b**2)
    ) / (2 * (m**2 + 1))
    y = m * x + b
    return x, y


def project_to_line(
    g: np.ndarray[float],
    s: np.ndarray[float],
    x: np.ndarray[float],
    y: np.ndarray[float],
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    # Create line segment
    v = np.array([np.diff(x), np.diff(y)])
    v_norm_sq = np.einsum("ij,ij", v, v)

    # Vectorize all coordinates
    p = np.stack([g.flatten(), s.flatten()], axis=1)

    # Distance from line start
    a = np.array([x[0], y[0]])
    d = p - a

    # Scalar projection
    t = d @ v / v_norm_sq

    # Reprojected coordinates
    p = a + np.outer(t, v)
    gp = p[:, 0].reshape(g.shape)
    sp = p[:, 1].reshape(s.shape)
    return gp, sp


def get_endpoints_from_projection(
    gp: np.ndarray[float],
    sp: np.ndarray[float],
    x: np.ndarray[float],
    y: np.ndarray[float],
    tau: np.ndarray[float],
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    # Get fraction and lifetime of projected points
    # Repeat arrays to calculate for two species
    gp = np.concatenate([gp, gp], axis=0)
    sp = np.concatenate([sp, sp], axis=0)
    x = np.expand_dims(x, axis=(1, 2))
    y = np.expand_dims(y, axis=(1, 2))

    # Distance formula from intersections (flip b/c distance from 2 -> alpha 1)
    alphas = np.sqrt((gp - np.flip(x, axis=0)) ** 2 + (sp - np.flip(y, axis=0)) ** 2)
    total = np.sum(alphas, axis=0)
    alphas /= total  # Ensure normalization

    # Weighted average for lifetimes
    while tau.ndim < alphas.ndim:
        tau = np.expand_dims(tau, axis=-1)
    tau_m = np.sum(tau * alphas, axis=0)

    return alphas, tau_m
