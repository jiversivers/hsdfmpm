import struct
from typing import Optional, Union, Tuple
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import Model, Data, ODR
from scipy.stats import f
from matplotlib.patches import Circle


def read_chars(f, count):
    r"""
    Read *count* bytes from a binary file object, decode with Latin-1, strip
    trailing NULs and return the resulting string.

    :param f: Open file handle positioned at the byte offset to read.
    :type  f: typing.BinaryIO
    :param count: Number of bytes to read.
    :type  count: int
    :return: Decoded, right-trimmed text.
    :rtype: str
    """
    return f.read(count).decode("latin-1").strip("\x00")


def read_fmt(f, fmt):
    r"""
    Convenience wrapper around :pyfunc:`struct.unpack`.

    Reads exactly ``struct.calcsize(fmt)`` bytes, unpacks them with the given
    *fmt* string (little-endian/host endian etc.) and returns **the first**
    element.  This matches the fact that most SDT header fields are scalar.

    :param f: Binary file object positioned at the field to decode.
    :type  f: typing.BinaryIO
    :param fmt: `struct` format string (e.g. ``"<H"`` for *uint16* little-endian).
    :type  fmt: str
    :return: Unpacked scalar value.
    :rtype: int | float
    """
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def continue_parsing_measurement_info(f, mi):
    r"""
    Continue reading ―in place― the **measurementInfo** block of a Becker & Hickl
    *\*.sdt* file.

    The function mutates *mi* by adding three nested dictionaries
    ``FCSInfo``, ``measurementInfoHISTInfo``, and
    ``measurementInfoHISTInfoExt`` plus several scalar fields that follow those
    sub-blocks.

    :param f: Binary file handle, already positioned at the first field of
        *FCSInfo* (channel number).
    :type  f: typing.BinaryIO
    :param mi: Partially-filled dictionary produced earlier in the parsing
        routine.
    :type  mi: dict[str, typing.Any]
    :return: The same dictionary instance (*mi*) now fully populated.
    :rtype: dict[str, typing.Any]
    """
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
    r"""
    Recursively convert *numpy* scalars, ``bytes`` and nested containers to
    plain Python types suitable for :pymod:`json`.

    * **dict** → dict with values converted
    * **numpy.integer / numpy.floating** → ``int`` / ``float`` via ``.item()``
    * **bytes** → decoded *utf-8* string (errors ignored)
    * **list / tuple** → same container type with converted items

    :param obj: Arbitrary object (deeply nested structures are handled).
    :type  obj: typing.Any
    :return: JSON-friendly equivalent of *obj*.
    :rtype: typing.Any
    """
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
    r"""
    Parse an entire Becker & Hickl **SDT/PLU** FLIM file and return the raw
    decay cube **plus** a rich metadata dictionary.

    The function reads, in order:

    #. *HeaderInfo* block
    #. *fileInformation* ASCII block
    #. *setup* ASCII block
    #. *measurementInfo* (and its sub-blocks)
    #. *decayInfo*
    #. photon count data (``uint16``) reshaped to
       ``(frames × Y × X × ADC)``

    :param filename: Path to the ``.sdt`` file on disk.
    :type  filename: str | pathlib.Path
    :return: Nested mapping with **headerInfo**, **fileInformation**, **setup**,
        **measurementInfo**, **decayInfo**, and the 4-D ``numpy.ndarray`` under
        the key ``"decay"``.
    :rtype: dict[str, typing.Any]
    """
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
    r"""
    Lightweight wrapper around :pyfunc:`open_sdt_file_full_with_metadata` that
    keeps only the *decay* array and a **JSON-serialisable** subset of the
    metadata.

    :param filename: Path to the ``.sdt`` file.
    :type  filename: str | pathlib.Path
    :return: Tuple ``(decay, meta)`` where *decay* is the 4-D photon cube
        and *meta* is a dict ready for ``json.dumps``.
    :rtype: tuple[numpy.ndarray, dict[str, typing.Any]]
    """
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
    x: NDArray[float], y: NDArray[float]
) -> tuple[NDArray[float], NDArray[float]]:
    r"""Get cartesian coordinates from polar coordinates.

    :param x: The `x` coordinates to convert.
    :type x: NDArray[float]
    :param y: The `y` coordinates to convert.
    :type y: NDArray[float]
    :return: The polar coordinates equivalent to the input cartesian coordinates.
    :rtype: tuple[NDArray[float], NDArray[float]]
    """
    z = x + 1j * y
    return np.angle(z), np.abs(z)


def cartesian_from_polar(
    theta: NDArray[float], r: NDArray[float], as_complex: bool = False
) -> Union[tuple[NDArray[float], NDArray[float]], NDArray[complex]]:
    r"""Get cartesian coordinates from polar coordinates as either (`x`,` y`) or a single complex number, `P`.

    :param theta: Polar Angle :math:`\theta`
    :type theta: NDArray[float]
    :param r: Polar radius :math:`r`
    :type r: NDArray[float]
    :param as_complex: If `True`, convert to complex before returning
    :return: The cartesian coordinates equivalent to the input polar coordinates.
    :rtype: tuple[NDArray[float], NDArray[float]] | NDArray[complex]
    """
    z = r * np.exp(1j * theta)
    if as_complex:
        return z
    return z.real, z.imag


def lifetime_from_cartesian(
    x: NDArray[float], y: NDArray[float], omega: float, harmonic: int = 1
) -> NDArray[float]:
    r"""Get the lifetime of the cartesian points queried.

    :param x: The x-coordinate, `G`.
    :type x: NDArray[float]
    :param y: The y-coordinate, `S`.
    :type y: NDArray[float]
    :param omega: Frequency of the excitation.
    :type omega: float
    :param harmonic: Harmonic of the phasro to get the coordinates in.
    :type harmonic: int
    :return: The lifetime at the points queried.
    :rtype: NDArray[float]
    """
    return (1 / omega) * np.tan(np.arctan2(y, x))


def cartesian_from_lifetime(
    tau: float, omega: float, harmonic: int = 1, as_complex: bool = False
) -> Union[Tuple[NDArray[float], NDArray[float]], NDArray[complex]]:
    r"""Get cartesian coordinates in phasor space from a lifetime.

    :param tau: Lifetime to get the polar coordinates from.
    :type tau: float
    :param omega: Frequency of the excitation.
    :type omega: float
    :param harmonic: Harmonic of the phasro to get the coordinates in.
    :type harmonic: int
    :param as_complex: If True, return a complex numpy array. Else, return a tuple of (real, imaginary).
    :type as_complex: bool
    :return: Cartesian coordinates (or complex number) in phasor space of the lifetime queried.
    :rtype: Union[Tuple[NDArray[float], NDArray[float]], NDArray[complex]]

    """
    phi, m = polar_from_lifetime(tau, omega, harmonic)
    return cartesian_from_polar(phi, m, as_complex)


def polar_from_lifetime(
    tau: NDArray[float], omega: float, harmonic: int = 1
) -> tuple[NDArray[float], NDArray[float]]:
    r"""Get polar coordinates from a lifetime in phasor space.

    :param tau: Lifetime to get the polar coordinates from.
    :type tau: float
    :param omega: Frequency of the excitation.
    :type omega: float
    :param harmonic: Harmonic of the phasro to get the coordinates in.
    :type harmonic: int
    :return: A tuple of polar coordinates, (:math:`\phi`, :math:`m`).
    :rtype: tuple[NDArray[float], NDArray[float]]
    """
    phi = np.arctan(harmonic * omega * tau)
    m = 1 / np.sqrt(1 + (harmonic * omega * tau) ** 2)
    return phi, m


def complex_phasor(g: NDArray[float], s: NDArray[float]) -> np.ndarray[complex]:
    r"""Convert phasor coordinates to single complex value.

    .. math:

    P = G + Si

    :param g: Real phasor coordinate (horizontal axis).
    :type g: numpy.ndarray[float]
    :param s: Imaginary phasor coordinate (vertical axis).
    :type s: numpy.ndarray[float]
    :returns: Complex phasor coordinate.
    :rtype: numpy.ndarray[complex]
    """
    return g + 1j * s


def plot_universal_circle(
    omega: float,
    harmonic: int = 1,
    tau_labels: Optional[list[float]] = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Axes, NDArray[float]]:
    r"""Helper function to plot a universal circle.
    :param omega: Frequency of the imaging excitation.
    :type omega: float
    :param harmonic: Harmonic of the circle to plot.
    :type harmonic: int
    :param tau_labels: Lifetieme labels to mark on the circle.
    :type tau_labels: list[float], optional
    :param ax: Axes to plot on.
    :type ax: matplotlib.axes.Axes, optional
    :return: Axes with cirlce and (x, y) coordinates of requested labels (that are marked on the circle).
    :rtype: tuple[matplotlib.axes.Axes, NDArray[float]]
    """
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


def _make_t_array(
    T: int,
    dt: float,
    ndim: int,
) -> NDArray[float]:
    """Helper to create an array of shape ... x :param:`T` (bin numbers) for each time frame of the decay.

    :param T: Total number of time steps.
    :type T: int
    :param dt: Time step.
    :type dt: float
    :param ndim: Number of dimensions to expand the array to.
    :type ndim: int
    :returns: Array increasing over axis -1 in increments of dt of shape ... x :param:`T`
    :rtype: NDArray[float]
    """
    t = np.expand_dims(np.arange(0.5, T, 1) * dt, axis=0)
    while t.ndim < ndim:
        t = np.expand_dims(t, axis=0)
    return t


def get_phasor_coordinates(
    decay: NDArray[float],
    bin_width: float = 10 / 256 / 1e9,
    frequency: float = 80e6,
    harmonic: int = 1,
    as_complex: bool = False,
) -> Union[
    tuple[NDArray[float], NDArray[float], NDArray[float]],
    tuple[NDArray[complex], NDArray[float]],
]:
    r"""Get the raw phasor coordinates from a decay curve given the imaging parameters provided.

    :param decay: The decay curve to get the coordinates from, shape CxHxWxT where C are color channels (if any) and T is the time bijn axis.
    :type decay: NDArray[float]
    :param bin_width: The temporal bin width to use, default 10 / 256 / 1e9
    :type bin_width: float
    :param frequency: The excitation frequency to use, default 80e6
    :type frequency: float
    :param harmonic: The harmonic to use, default 1
    :type harmonic: int
    :param as_complex: If true, convert to complex before returning
    :type as_complex: bool
    :return: The raw phasor coordinates and the voxel-wise photon counts. If :param:`as_complex` is true, raw
     coordinates are returned as a single array of complex numbers.
    :rtype: tuple[NDArray[float], NDArray[float],NDArray[float]] | tuple[NDArray[complex], NDArray[float]]
    """
    w = 2 * np.pi * harmonic * frequency
    t = _make_t_array(decay.shape[-1], bin_width, decay.ndim)

    # Calculate raw phasor coordinates
    g = np.sum(decay * np.cos(w * t), axis=-1)
    s = np.sum(decay * np.sin(w * t), axis=-1)
    photons = np.sum(decay, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        g /= photons
        s /= photons

    if as_complex:
        return complex_phasor(g, s), photons
    return g, s, photons


def fit_phasor(
    g: NDArray[float],
    s: NDArray[float],
) -> dict[str, float]:
    r"""
    Perform a **total least–squares** (TLS) line fit on a phasor point-cloud.

    The real coordinates *g* (horizontal axis) and imaginary coordinates *s*
    (vertical axis) are jointly fitted with

    .. math::

        S = m\,G + b ,

    using :py:class:`scipy.odr.ODR`.
    After fitting the function

    #. **Whitens** the centered cloud by the variances of the fitted parameters,
    #. Computes the **aspect ratio** of the whitened cloud via SVD (:py:func:`~phasor_svd_aspect_ratio`),
    #. Evaluates an **F-statistic**

       .. math::

          F_{\text{obs}} = \frac{(N-2)\,\Sigma}{N},

       where :math:`\Sigma` is the aspect ratio and *N* the number of points,
    #. Reports the associated *p-value* (H₀: ``aspect_ratio == 1``).

    Parameters
    ----------
    :param g: Real phasor coordinates (``G`` axis).
    :type g: NDArray[float]

    :param s: Imaginary phasor coordinates (``S`` axis).
    :type s: NDArray[float]

    :returns: Mapping with the TLS result and derived statistics:
        ``fit_y_intercept``
            Y-intercept :math:`b` of the fitted line.

        ``fit_slope``
            Slope :math:`m` of the fitted line.

        ``aspect_ratio``
            Ratio of singular values (major/minor axis) of the whitened cloud;
            values » 1 indicate pronounced elongation.

        ``red_chi_squared``
            Reduced :math:`\chi^{2}` of the TLS fit
            (``odr_output.res_var``).

        ``p_value``
            *p-value* of the observed F-statistic (H₀: isotropic cloud).

        ``n``
            Number of non-NaN points used in the fit.
    :rtype: dict[str, float]
    """
    mask = ~(np.isnan(g) | np.isnan(s))

    # Make a simple linear model
    def phasor_line(beta, x):
        """Linear function: S as a function of G"""
        return beta[0] * x + beta[1]

    # Get TLS line
    odr = ODR(data=Data(g[mask], s[mask]), model=Model(phasor_line), beta0=[-0.5, 1])
    output = odr.run()
    m, b = output.beta

    # Get variance estimates and normalize coords to those to get whitened aspect ratio
    sg, ss = output.cov_beta[0, 0], output.cov_beta[1, 1]
    g_white, s_white = (g[mask] - g[mask].mean()) / sg, (s[mask] - s[mask].mean()) / ss
    ratio = phasor_svd_aspect_ratio(g_white, s_white)

    # Get statistics for non-unity aspect ratio of the normalized coords
    N = np.size(g[mask])
    F_obs = (N - 2) * ratio / N  # Observed F stat
    p_value = f.sf(F_obs, 2, N - 3)  # p-value of F statistic

    output = dict(
        fit_y_intercept=b,
        fit_slope=m,
        aspect_ratio=ratio,
        red_chi_squared=output.res_var,
        p_value=p_value,
        n=N,
    )

    return output


def phasor_svd_aspect_ratio(
    g: NDArray[float], s: NDArray[float]
) -> tuple[np.ndarray[float, float], float, np.ndarray[float, float]]:
    r"""Perform SVD on the point cloud coordinates described by :param:`g` and :param:`s`.

    :param g: Real phasor coordinates (horizontal axis points).
    :type g: NDArray[float]
    :param s: Imaginary phasor coordinates (vertical axis points).
    :type s: NDArray[float]
    :return: The SVD-based aspect ratio of the input cloud.
    :rtype: float
    """
    cloud = np.stack([g.flatten(), s.flatten()], axis=1)
    _, s, _ = np.linalg.svd(cloud, full_matrices=False)
    return s[0] / s[1]


def find_intersection_with_circle(
    b: float, m: float
) -> tuple[np.ndarray[float, float], np.ndarray[float, float]]:
    r"""
    Calculate the intersection of a line described by :param:`b` and :param:`m` with the universal circle (circle with
    center (0.5, 0) and radius 0.5).

    :param b: y-intercept of the query line
    :type b: float
    :param m: slope of the query line
    :type m: float
    :return: Tuple of (x-coordinates, y-coordinates) of the intersection points of the query line
    :rtype: tuple[np.ndarray[float, float], np.ndarray[float, float]]
    """
    # Calculate intersection points of the line (quadratic eqn)
    x = (
        -(2 * m * b - 1)
        + np.array([1, -1]) * np.sqrt((2 * m * b - 1) ** 2 - 4 * (m**2 + 1) * b**2)
    ) / (2 * (m**2 + 1))
    y = m * x + b
    return x, y


def project_to_line(
    g: NDArray[float],
    s: NDArray[float],
    x: NDArray[float],
    y: NDArray[float],
) -> tuple[NDArray[float], NDArray[float]]:
    r"""
    Project all points described by the ordered pairs (:param:`g:`, :param:`s`) onto the line described by the two
    ordered pairs (:param:`x:`, :param:`y:`).

    :param g: Real phasor coordinates (vertical axis points).
    :type g: NDArray[float]
    :param s: Imaginary phasor coordinates (vertical axis points).
    :type s: NDArray[float]
    :param x: x-coordinates of the line points.
    :type x: NDArray[float]
    :param y: y-coordinates of the line points.
    :type y: NDArray[float]
    :return: Tuple of (x-coordinates, y-coordinates) of the resulting projected points' locations
    """
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
    gp: NDArray[float],
    sp: NDArray[float],
    x: NDArray[float],
    y: NDArray[float],
    tau: NDArray[float],
) -> tuple[NDArray[float], NDArray[float]]:
    r"""
    Calculate the lifetime metric endpoints of the line described by the (projected) points in :param:`gp` and :param:`sp`
    with the :math:`\alpha` determiend by the distances from the univsersal circle intersection points give by :param:`x`
     and :param:`y`.

    :param gp: (Projected) real phasor coordinates (horizontal axis points).
    :type gp: NDArray[float]
    :param sp: (Projected) imaginary phasor coordinates (vertical axis points).
    :type sp: NDArray[float]
    :param x: x-coordinates of the circle-intersection points.
    :type x: NDArray[float]
    :param y: y-coordinates of the circle-intersection points.
    :type y: NDArray[float]
    :param tau: Lifetimes associated with each (x, y) point. Used to calcualte mean lifetime.
    :type tau: NDArray[float]
    :return: Tuple of (alphas, mean_lifetime) based on the distance of :param:`gp` and :param:`sp` from and value of :param:`x` and :param:`y` as given by :param:`tau`.
    :rtype: tuple[NDArray[float], NDArray[float]]
    """
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
