import glob
import os
import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..mpm.mpm import LifetimeImage, InstrumentResponseFunction


def read_chars(f, count):
    return f.read(count).decode('latin-1').strip('\x00')

def read_fmt(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

def continue_parsing_measurement_info(f, mi):
    # FCS Info
    fcs = {}
    fcs['chan'] = read_fmt(f, '<H')
    fcs['fcs_decay_calc'] = read_fmt(f, '<H')
    fcs['mt_resol'] = read_fmt(f, '<I')
    fcs['cortime'] = read_fmt(f, '<f')
    fcs['calc_photons'] = read_fmt(f, '<I')
    fcs['fcs_points'] = read_fmt(f, '<i')
    fcs['end_time'] = read_fmt(f, '<f')
    fcs['overruns'] = read_fmt(f, '<H')
    fcs['fcs_type'] = read_fmt(f, '<H')
    fcs['cross_chan'] = read_fmt(f, '<H')
    fcs['mod'] = read_fmt(f, '<H')
    fcs['cross_mod'] = read_fmt(f, '<H')
    fcs['cross_mt_resol'] = read_fmt(f, '<I')
    mi['FCSInfo'] = fcs

    mi['image_x'] = read_fmt(f, '<i')
    mi['image_y'] = read_fmt(f, '<i')
    mi['image_rx'] = read_fmt(f, '<i')
    mi['image_ry'] = read_fmt(f, '<i')
    mi['xy_gain'] = read_fmt(f, '<h')
    mi['dig_flags'] = read_fmt(f, '<h')
    mi['adc_de'] = read_fmt(f, '<h')
    mi['det_type'] = read_fmt(f, '<h')
    mi['x_axis'] = read_fmt(f, '<h')

    # HIST Info
    hist = {}
    hist['fida_time'] = read_fmt(f, '<f')
    hist['filda_time'] = read_fmt(f, '<f')
    hist['fida_points'] = read_fmt(f, '<i')
    hist['filda_points'] = read_fmt(f, '<i')
    hist['mcs_time'] = read_fmt(f, '<f')
    hist['mcs_points'] = read_fmt(f, '<i')
    hist['cross_calc_phot'] = read_fmt(f, '<I')
    hist['mcsta_points'] = read_fmt(f, '<H')
    hist['mcsta_flags'] = read_fmt(f, '<H')
    hist['mcsta_tpp'] = read_fmt(f, '<I')
    hist['calc_markers'] = read_fmt(f, '<I')
    hist['fcs_calc_phot'] = read_fmt(f, '<I')
    hist['reserved3'] = read_fmt(f, '<I')
    mi['measurementInfoHISTInfo'] = hist

    # HIST Info Extension
    hist_ext = {}
    hist_ext['first_frame_time'] = read_fmt(f, '<f')
    hist_ext['frame_time'] = read_fmt(f, '<f')
    hist_ext['line_time'] = read_fmt(f, '<f')
    hist_ext['pixel_time'] = read_fmt(f, '<f')
    hist_ext['scan_type'] = read_fmt(f, '<h')
    hist_ext['skip_2nd_line_clk'] = read_fmt(f, '<h')
    hist_ext['right_border'] = read_fmt(f, '<I')
    hist_ext['info'] = read_chars(f, 40)
    mi['measurementInfoHISTInfoExt'] = hist_ext

    mi['sync_delay'] = read_fmt(f, '<f')
    mi['sdel_ser_no'] = read_fmt(f, '<H')
    mi['sdel_input'] = read_fmt(f, 'B')
    mi['mosaic_ctrl'] = read_fmt(f, 'B')
    mi['mosaic_x'] = read_fmt(f, 'B')
    mi['mosaic_y'] = read_fmt(f, 'B')
    mi['frames_per_el'] = read_fmt(f, '<h')
    mi['chan_per_el'] = read_fmt(f, '<h')
    mi['mosaic_cycles_done'] = read_fmt(f, '<i')
    mi['mla_ser_no'] = read_fmt(f, '<H')
    mi['DCC_in_use'] = read_fmt(f, 'B')
    mi['dcc_ser_no'] = read_chars(f, 12)
    mi['TiSaLas_status'] = read_fmt(f, '<H')
    mi['TiSaLas_wav'] = read_fmt(f, '<H')
    mi['AOM_status'] = read_fmt(f, 'B')
    mi['AOM_power'] = read_fmt(f, 'B')
    mi['ddg_ser_no'] = read_chars(f, 8)
    mi['prior_ser_no'] = read_fmt(f, '<i')
    mi['mosaic_x_hi'] = read_fmt(f, 'B')
    mi['mosaic_y_hi'] = read_fmt(f, 'B')
    mi['reserve'] = read_chars(f, 12)

    return mi

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    return obj

def open_sdt_file_full_with_metadata(filename):
    with open(filename, 'rb') as f:
        out = {}

        out['headerInfo'] = {
            'revision': read_fmt(f, '<h'),
            'info_offset': read_fmt(f, '<l'),
            'info_length': read_fmt(f, '<h'),
            'setup_offs': read_fmt(f, '<l'),
            'setup_length': read_fmt(f, '<h'),
            'data_block_offset': read_fmt(f, '<l'),
            'no_of_data_blocks': read_fmt(f, '<h'),
            'data_block_length': read_fmt(f, '<L'),
            'meas_desc_block_offset': read_fmt(f, '<l'),
            'no_of_meas_desc_blocks': read_fmt(f, '<h'),
            'meas_desc_block_length': read_fmt(f, '<h'),
            'header_valid': read_fmt(f, '<H'),
            'reserved1': read_fmt(f, '<L'),
            'reserved2': read_fmt(f, '<H'),
            'chksum': read_fmt(f, '<H'),
        }

        out['fileInformation'] = f.read(out['headerInfo']['info_length']).decode('latin-1')
        out['setup'] = f.read(out['headerInfo']['setup_length']).decode('latin-1')

        # Measurement Info
        mi = {}
        mi['time'] = read_chars(f, 9)
        mi['date'] = read_chars(f, 11)
        mi['mod_ser_no'] = read_chars(f, 16)
        mi['measurementInfo_mode'] = read_fmt(f, '<h')
        mi['cfd_ll'] = read_fmt(f, '<f')
        mi['cfd_lh'] = read_fmt(f, '<f')
        mi['cfd_zc'] = read_fmt(f, '<f')
        mi['cfd_hf'] = read_fmt(f, '<f')
        mi['syn_zc'] = read_fmt(f, '<f')
        mi['syn_fd'] = read_fmt(f, '<h')
        mi['syn_hf'] = read_fmt(f, '<f')
        mi['tac_r'] = read_fmt(f, '<f')
        mi['tac_g'] = read_fmt(f, '<h')
        mi['tac_of'] = read_fmt(f, '<f')
        mi['tac_ll'] = read_fmt(f, '<f')
        mi['tac_lh'] = read_fmt(f, '<f')
        mi['adc_re'] = read_fmt(f, '<h')
        mi['eal_de'] = read_fmt(f, '<h')
        mi['ncx'] = read_fmt(f, '<h')
        mi['ncy'] = read_fmt(f, '<h')
        mi['page'] = read_fmt(f, '<H')
        mi['col_t'] = read_fmt(f, '<f')
        mi['rep_t'] = read_fmt(f, '<f')
        mi['stopt'] = read_fmt(f, '<h')
        mi['overfl'] = read_fmt(f, 'B')
        mi['use_motor'] = read_fmt(f, '<h')
        mi['steps'] = read_fmt(f, '<H')
        mi['offset'] = read_fmt(f, '<f')
        mi['dither'] = read_fmt(f, '<h')
        mi['incr'] = read_fmt(f, '<h')
        mi['mem_bank'] = read_fmt(f, '<h')
        mi['mod_type'] = read_chars(f, 16)
        mi['syn_th'] = read_fmt(f, '<f')
        mi['dead_time_comp'] = read_fmt(f, '<h')
        mi['polarity_l'] = read_fmt(f, '<h')
        mi['polarity_f'] = read_fmt(f, '<h')
        mi['polarity_p'] = read_fmt(f, '<h')
        mi['linediv'] = read_fmt(f, '<h')
        mi['accumulate'] = read_fmt(f, '<h')
        mi['flbck_y'] = read_fmt(f, '<i')
        mi['flbck_x'] = read_fmt(f, '<i')
        mi['bord_u'] = read_fmt(f, '<i')
        mi['bord_l'] = read_fmt(f, '<i')
        mi['pix_time'] = read_fmt(f, '<f')
        mi['pix_clk'] = read_fmt(f, '<h')
        mi['trigger'] = read_fmt(f, '<h')
        mi['scan_x'] = read_fmt(f, '<i')
        mi['scan_y'] = read_fmt(f, '<i')
        mi['scan_rx'] = read_fmt(f, '<i')
        mi['scan_ry'] = read_fmt(f, '<i')
        mi['fifo_typ'] = read_fmt(f, '<h')
        mi['epx_div'] = read_fmt(f, '<i')
        mi['mod_type_code'] = read_fmt(f, '<H')
        mi['mod_fpga_ver'] = read_fmt(f, '<H')
        mi['overflow_corr_factor'] = read_fmt(f, '<f')
        mi['adc_zoom'] = read_fmt(f, '<i')
        mi['cycles'] = read_fmt(f, '<i')

        # StopInfo block
        stop = {
            'status': read_fmt(f, '<H'),
            'flags': read_fmt(f, '<H'),
            'stop_time': read_fmt(f, '<f'),
            'cur_step': read_fmt(f, '<i'),
            'cur_cycle': read_fmt(f, '<i'),
            'cur_page': read_fmt(f, '<i'),
            'min_sync_rate': read_fmt(f, '<f'),
            'min_cfd_rate': read_fmt(f, '<f'),
            'min_tac_rate': read_fmt(f, '<f'),
            'min_adc_rate': read_fmt(f, '<f'),
            'max_sync_rate': read_fmt(f, '<f'),
            'max_cfd_rate': read_fmt(f, '<f'),
            'max_tac_rate': read_fmt(f, '<f'),
            'max_adc_rate': read_fmt(f, '<f'),
            'reserved1': read_fmt(f, '<i'),
            'reserved2': read_fmt(f, '<f'),
        }
        mi['StopInfo'] = stop

        mi = continue_parsing_measurement_info(f, mi)
        out['measurementInfo'] = mi

        # Decay block
        out['decayInfo'] = {
            'data_offs_ext': read_fmt(f, '<B'),
            'next_block_offs_ext': read_fmt(f, '<B'),
            'data_offs': read_fmt(f, '<L'),
            'next_block_offs': read_fmt(f, '<L'),
            'block_type': read_fmt(f, '<H'),
            'meas_desc_block_no': read_fmt(f, '<h'),
            'lblock_no': read_fmt(f, '<L'),
            'block_length': read_fmt(f, '<L'),
        }

        # Decay data
        decay_data = np.fromfile(f, dtype=np.uint16)
        ny, nx, adc = mi['image_y'], mi['image_x'], mi['adc_re']
        decay = decay_data[:nx * ny * adc].reshape((adc, nx, ny)).transpose(2, 1, 0)
        out['decay'] = {'Channel_1': decay}

    return out

def open_sdt_file_with_json_metadata(filename):
    out = open_sdt_file_full_with_metadata(filename)
    json_metadata = make_json_serializable({
        "headerInfo": out["headerInfo"],
        "fileInformation": out["fileInformation"],
        "setup": out["setup"],
        "measurementInfo": out["measurementInfo"],
        "decayInfo": out["decayInfo"],
    })
    return out["decay"], json_metadata

def get_irf(irf: Optional[Union['InstrumentResponseFunction', str]] = None) -> 'InstrumentResponseFunction':
    """
    This is a helper function to parse the input IRF as either a file, a preloaded LifetimeImage, or None. In the case
    of None, the default path is from in .hsdfm package path. This is searched and the most recent IRF filepath is used.
    In this case and the case of an input filepath, the .sdt file is loaded into a LifetimeImage object and returned. In
    the case of an input LifetimeImage object, it is simply returned.

    :param irf: The path to the IRF file, or a preloaded IRF in a LifetimeImage object. Optional.
    :type irf: Union['LifetimeImage', str]

    :raises FileNotFoundError: If no file is selected when required.
    :raises ValueError: If the IRF file format is unsupported or corrupt.

    :return: A LifetimeImage object for the IRF file.
    :rtype: LifetimeImage

    """
    if isinstance(irf, 'InstrumentResponseFunction'):
        return irf

    irf_file = None
    if irf is None:
        # Check for saved IRF
        irf_path = Path.home() / ".hsdfmpm"
        irf_file = glob.glob(os.path.join(irf_path, "*.irf"))[0]
        irf_file = glob.glob(os.path.join(irf_path, f'*.sdt'))[0]
        # Check for the IRF file
        if not irf_file:
            raise FileNotFoundError(f"No IRF file found in {irf_path}")
    elif isinstance(irf, str):
        irf_file = Path(irf)
    if not os.path.isfile(irf_file):
        raise FileNotFoundError(f"No IRF file found at {irf_file}")

    if irf_file is not None:
        # Load input IRF
        irf = InstrumentResponseFunction.load(image_path=irf)

    return irf

def calibrate_irf(irf: LifetimeImage, save: bool = True, **kwargs):
    """
    Calibrates phasor parameters using an input IRF (Instrument Response Function).

    This function computes phasor calibration parameters from an IRF file, which may be
    a `.tiff` stack, a raw `.sdt` file, or a path to one of these file types. If no input
    is provided, a GUI file selector will prompt the user to choose a file.

    The resulting calibration is stored in a `params` structure within the returned
    `LifetimeImage` object. This structure includes laser and timing information, as well
    as calculated phasor calibration values (modulation and phase), both as pixel-wise
    maps and global means.

    :param irf: Instrument Response Function image or file path to be used for calibration.
                If left empty, a GUI will prompt for file selection.
    :type irf: LifetimeImage

    :return: A LifetimeImage object with an updated `params` field containing the calibration data.
    :rtype: LifetimeImage

    Calibration fields included in `params`:

    - ``f`` (float): Laser frequency in MHz.
    - ``dt`` (float): Temporal bin width in nanoseconds.
    - ``n`` (int): Harmonic number.
    - ``calibration`` (dict): Nested dictionary with the following keys:
        - ``tau_ref`` (float): Reference lifetime in ns.
        - ``m_ref`` (float, optional): Reference modulation.
        - ``phi_ref`` (float, optional): Reference phase.
        - ``w`` (float): Angular laser frequency in rad/s.
        - ``T`` (int): Number of time bins.
        - ``Map`` (dict): Pixel-wise calibration values.
        - ``Mean`` (dict): Mean calibration values.

    Saving behavior is controlled by a SAVEOPT argument or a UI popup:

    - Save: 'on', 'save', True, 1
    - Ask: 'ask', ''
    - Do not save: 'off', 'nosave', False, 0

    Additional keyword arguments are passed to ``getPhasorCoords``. Refer to
    ``getPhasorCoords`` and ``SDTtoTimeStack`` for supported options.
    :param irf:
    :return:
    """

    period = irf.metadata['']
