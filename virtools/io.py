"""I/O handlers for LBL, QUB, and housekeeping data."""
import pvl
import numpy as np
import sys
import os
from datetime import datetime
# from osgeo import gdal

def parse_lbl_metadata(lbl_path):
    lbl_data = pvl.load(lbl_path)
    qube = lbl_data.get("QUBE", {})
    bands, samples, lines = qube.get("CORE_ITEMS", [None]*3)
    item_bytes = qube.get("CORE_ITEM_BYTES")
    item_type = qube.get("CORE_ITEM_TYPE")
    dtype_map = {
        (2, "MSB_INTEGER"): ">i2",
        (2, "LSB_INTEGER"): "<i2",
        (4, "MSB_INTEGER"): ">i4",
        (4, "LSB_INTEGER"): "<i4",
        (4, "IEEE_REAL"): ">f4",
        (8, "IEEE_REAL"): ">f8",
    }
    dtype = dtype_map.get((item_bytes, item_type))
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {item_bytes} {item_type}")
    band_bin = qube.get("BAND_BIN", {})
    def parse_time(key):
        val = lbl_data.get(key)
        if isinstance(val, str):
            try:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        return val

    def get_float(key, default=np.nan):
        try:
            return float(lbl_data.get(key, default))
        except Exception:
            return default

    def get_str(key, default=""):
        val = lbl_data.get(key)
        return str(val).strip() if val else default
    exposure_time = None
    external_repetition_time = None
    if "FRAME_PARAMETER" in lbl_data and "FRAME_PARAMETER_DESC" in lbl_data:
        try:
            param_desc = lbl_data["FRAME_PARAMETER_DESC"]
            param_values = lbl_data["FRAME_PARAMETER"]
            idx_exp = param_desc.index("EXPOSURE_DURATION")
            exposure_time = float(param_values[idx_exp])
            idx_ert = param_desc.index("EXTERNAL_REPETITION_TIME")
            external_repetition_time = float(param_values[idx_ert])
        except Exception:
            pass

    return {
        "shape": (bands, samples, lines),
        "dtype": dtype,
        "core_null": qube.get("CORE_NULL", -32768),
        "core_low_saturation": qube.get("CORE_LOW_REPR_SATURATION", -32767),
        "core_high_saturation": qube.get("CORE_HIGH_REPR_SATURATION", -32767),
        "core_multiplier": qube.get("CORE_MULTIPLIER", 1.0),
        "core_base": qube.get("CORE_BASE", 0.0),
        "product_type": lbl_data.get("PRODUCT_TYPE", "UNKNOWN"),
        "wave_length_cen": np.array(band_bin.get("BAND_BIN_CENTER", []), dtype=np.float32),
        "wave_width": np.array(band_bin.get("BAND_BIN_WIDTH", []), dtype=np.float32),
        "wave_length_band_val": np.array(band_bin.get("BAND_BIN_ORIGINAL_BAND", []), dtype=np.int16),
        "spacecraft_solar_dist": float(qube.get("SPACECRAFT_SOLAR_DISTANCE", 441765159.0)),
        "start_time": parse_time("START_TIME"),
        "end_time": parse_time("STOP_TIME"),
        "exposure_time": exposure_time,
        "external_repetition_time": external_repetition_time,
        "mission_phase": get_str("MISSION_PHASE_NAME"),
        "solar_incidence": get_float("INCIDENCE_ANGLE"),
        "emission_angle": get_float("EMISSION_ANGLE"),
        "phase_angle": get_float("PHASE_ANGLE"),
        "sc_target_distance": get_float("TARGET_CENTER_DISTANCE"),
        "sub_spacecraft_lat": get_float("SUB_SPACECRAFT_LATITUDE"),
        "sub_spacecraft_lon": get_float("SUB_SPACECRAFT_LONGITUDE"),
        "local_hour_angle": get_float("LOCAL_HOUR_ANGLE"),
        "target_name": get_str("TARGET_NAME"),
        "target_type": get_str("TARGET_TYPE"),
    }

def fix_endianness(array, dtype):
    dtype = np.dtype(dtype)
    if dtype.byteorder not in ('=', '|') and dtype.byteorder != sys.byteorder:
        array = array.byteswap().view(dtype.newbyteorder())
    return array
def load_qub_from_lbl(lbl_path, qub_path, cal=False, return_flag_mask=False):
    meta = parse_lbl_metadata(lbl_path)
    shape = meta["shape"]
    dtype = meta["dtype"]
    with open(qub_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    data = fix_endianness(data, dtype).reshape(shape, order="F").astype(np.float32)
    null, low, high = meta["core_null"], meta["core_low_saturation"], meta["core_high_saturation"]
    flag_mask = np.zeros_like(data, dtype=np.uint8)  # 0 = valid
    flag_mask[data == null] = 1  # null
    flag_mask[data == low] = 2  # low saturation
    flag_mask[data == high] = 3  # high saturation
    # flag_mask[data == -32766] = 4
    # flag_mask[data == -32765] = 5
    # flag_mask[data == -32764] = 6
    # flag_mask[data == -32762] = 7
    # flag_mask[data == -32761] = 8
    data = data.astype(np.float32)
    data[flag_mask != 0] = np.nan
    if not cal:
        mult, base = meta["core_multiplier"], meta["core_base"]
        if mult != 1.0 or base != 0.0:
            data *= mult
            data += base
    if return_flag_mask:
        return data, meta, flag_mask
    else:
        return data, meta

def load_qub_from_lbl_name(base_folder, lbl_filename, cal=False, return_flag_mask=False):
    lbl_path = os.path.join(base_folder, lbl_filename)
    lbl_data = pvl.load(lbl_path)
    qube = lbl_data.get("QUBE", {})
    bands, samples, lines = qube.get("CORE_ITEMS", [None]*3)
    item_bytes = qube.get("CORE_ITEM_BYTES")
    item_type = qube.get("CORE_ITEM_TYPE")

    dtype_map = {
        (2, "MSB_INTEGER"): ">i2",
        (2, "LSB_INTEGER"): "<i2",
        (4, "MSB_INTEGER"): ">i4",
        (4, "LSB_INTEGER"): "<i4",
        (4, "IEEE_REAL"): ">f4",
        (8, "IEEE_REAL"): ">f8",
    }
    dtype = dtype_map.get((item_bytes, item_type))
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {item_bytes} {item_type}")

    def fix_endianness(array, dtype):
        dtype = np.dtype(dtype)
        if dtype.byteorder not in ('=', '|') and dtype.byteorder != sys.byteorder:
            array = array.byteswap().view(dtype.newbyteorder())
        return array

    def parse_time(key):
        val = lbl_data.get(key)
        if isinstance(val, str):
            try:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        return val

    def get_float(key, default=np.nan):
        try:
            return float(lbl_data.get(key, default))
        except Exception:
            return default

    def get_str(key, default=""):
        val = lbl_data.get(key)
        return str(val).strip() if val else default

    band_bin = qube.get("BAND_BIN", {})
    exposure_time = None
    external_repetition_time = None
    if "FRAME_PARAMETER" in lbl_data and "FRAME_PARAMETER_DESC" in lbl_data:
        try:
            param_desc = lbl_data["FRAME_PARAMETER_DESC"]
            param_values = lbl_data["FRAME_PARAMETER"]
            idx_exp = param_desc.index("EXPOSURE_DURATION")
            exposure_time = float(param_values[idx_exp])
            idx_ert = param_desc.index("EXTERNAL_REPETITION_TIME")
            external_repetition_time = float(param_values[idx_ert])
        except Exception:
            pass

    meta = {
        "shape": (bands, samples, lines),
        "dtype": dtype,
        "core_null": qube.get("CORE_NULL", -32768),
        "core_low_saturation": qube.get("CORE_LOW_REPR_SATURATION", -32767),
        "core_high_saturation": qube.get("CORE_HIGH_REPR_SATURATION", -32767),
        "core_multiplier": qube.get("CORE_MULTIPLIER", 1.0),
        "core_base": qube.get("CORE_BASE", 0.0),
        "product_type": lbl_data.get("PRODUCT_TYPE", "UNKNOWN"),
        "wave_length_cen": np.array(band_bin.get("BAND_BIN_CENTER", []), dtype=np.float32),
        "wave_width": np.array(band_bin.get("BAND_BIN_WIDTH", []), dtype=np.float32),
        "wave_length_band_val": np.array(band_bin.get("BAND_BIN_ORIGINAL_BAND", []), dtype=np.int16),
        "spacecraft_solar_dist": float(qube.get("SPACECRAFT_SOLAR_DISTANCE", 441765159.0)),
        "start_time": parse_time("START_TIME"),
        "end_time": parse_time("STOP_TIME"),
        "exposure_time": exposure_time,
        "external_repetition_time": external_repetition_time,
        "mission_phase": get_str("MISSION_PHASE_NAME"),
        "solar_incidence": get_float("INCIDENCE_ANGLE"),
        "emission_angle": get_float("EMISSION_ANGLE"),
        "phase_angle": get_float("PHASE_ANGLE"),
        "sc_target_distance": get_float("TARGET_CENTER_DISTANCE"),
        "sub_spacecraft_lat": get_float("SUB_SPACECRAFT_LATITUDE"),
        "sub_spacecraft_lon": get_float("SUB_SPACECRAFT_LONGITUDE"),
        "local_hour_angle": get_float("LOCAL_HOUR_ANGLE"),
        "target_name": get_str("TARGET_NAME"),
        "target_type": get_str("TARGET_TYPE"),
    }
    qub_relative = lbl_data.get("^QUBE")
    if isinstance(qub_relative, str):
        qub_path = os.path.join(base_folder, qub_relative.strip('"'))
    else:
        raise ValueError("Could not find QUBE path in label.")
    with open(qub_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    data = fix_endianness(data, dtype).reshape(meta["shape"], order="F").astype(np.float32)
    null, low, high = meta["core_null"], meta["core_low_saturation"], meta["core_high_saturation"]
    flag_mask = np.zeros_like(data, dtype=np.uint8)  # 0 = valid
    flag_mask[data == null] = 1  # null
    flag_mask[data == low] = 2  # low saturation
    flag_mask[data == high] = 3  # high saturation
    data[flag_mask != 0] = np.nan
    if not cal:
        mult, base = meta["core_multiplier"], meta["core_base"]
        if mult != 1.0 or base != 0.0:
            data *= mult
            data += base

    if return_flag_mask:
        return data, meta, flag_mask
    else:
        return data, meta

def extract_hk_data(lbl_hk_file, tab_hk_file):
    lbl_data = pvl.load(lbl_hk_file)
    columns = lbl_data["TABLE"].getlist("COLUMN")
    # Map PDS column names to internal keys
    name_map = {
        "SHUTTER STATUS": "shutter_status",
        "IR EXPO": "exposure_time_ir",
        "IR TEMP": "ir_temp",
        "CCD EXPO": "exposure_time_ccd",
        "CCD TEMP": "ccd_temp",
        "SPECT TEMP": "spect_temp",
        "TELE TEMP": "tele_temp",
        "COLD TIP TEMP": "cold_tip_temp",
        "RADIATOR TEMP": "radiator_temp",
        "LEDGE TEMP": "ledge_temp",
        "MIRROR SIN": "mirror_sin",
        "MIRROR COS": "mirror_cos",
        "START NOISY BITS": "start_noisy_bits",
        "END NOISY BITS": "end_noisy_bits",
        "NOF NOISY BITS": "num_noisy_bits",
        "CR ROW": "cr_row",
        "SUBFRAME DATA": "subframe_data",
        "SEQ STEP": "seq_step",
    }
    column_specs = {}
    for col in columns:
        name = col["NAME"].strip().upper()
        if name in name_map:
            key = name_map[name]
            start = int(col["START_BYTE"]) - 1
            length = int(col["BYTES"])
            column_specs[key] = slice(start, start + length)

    missing = set(name_map.values()) - set(column_specs.keys())
    if missing:
        print(f"Warning: Missing columns in LBL: {missing}")
    extracted_data = {key: [] for key in column_specs}
    with open(tab_hk_file) as f:
        lines = f.readlines()
    shutter_open = {"open": 1, "closed": 0}
    for line in lines:
        for key, sl in column_specs.items():
            raw = line[sl].strip()
            if key == "shutter_status":
                extracted_data[key].append(shutter_open.get(raw.lower(), None))
            else:
                try:
                    if raw.isdigit():
                        extracted_data[key].append(int(raw))
                    else:
                        extracted_data[key].append(float(raw))
                except ValueError:
                    extracted_data[key].append(None)

    shutter_data = extracted_data.get("shutter_status", [])
    closed_indexes = [i for i, val in enumerate(shutter_data) if val == 0]
    opened_indexes = [i for i, val in enumerate(shutter_data) if val == 1]

    return {
        "data": extracted_data,
        "closed_indexes": closed_indexes,
        "opened_indexes": opened_indexes,}

def load_ITF_data(filename, shape=(432, 256), dtype=np.float64):
    data = np.fromfile(filename, dtype=dtype).reshape(shape, order='F')
    return data.reshape(shape)

def parse_pds3_label(lbl_path):
    """
    Parse a PDS3 .LBL file and extract metadata for EACH TABLE object.

    Returns a list of tables, each a dict:
      {
        'pointer':   str or None     # the "^TABLE" filename, if present
        'record_bytes': int          # bytes per record (ROW_BYTES or fallback to RECORD_BYTES)
        'file_records': int          # number of rows    (ROWS     or fallback to FILE_RECORDS)
        'columns': [
            {
              'name':       str,
              'data_type':  str,
              'start_byte': int,
              'bytes':      int
            },
            ...
        ]
      }
    """
    lbl = pvl.load(lbl_path)
    pointers = lbl.get('^TABLE')
    if isinstance(pointers, str):
        pointers = [pointers]
    elif pointers is None:
        pointers = []
    tables = lbl.get('TABLE')
    if isinstance(tables, dict):
        tables = [tables]
    elif tables is None:
        raise ValueError("No TABLE object found in label")
    if len(pointers) < len(tables):
        pointers = pointers + [None] * (len(tables) - len(pointers))
    out = []
    for ptr, tbl in zip(pointers, tables):
        rec_bytes = tbl.get('ROW_BYTES', lbl.get('RECORD_BYTES'))
        row_cnt   = tbl.get('ROWS',     lbl.get('FILE_RECORDS'))
        if rec_bytes  is None or row_cnt is None:
            raise ValueError("Missing ROW_BYTES/RECORD_BYTES or ROWS/FILE_RECORDS")
        cols = tbl.get('COLUMN')
        if isinstance(cols, dict):
            cols = [cols]
        col_defs = []
        for c in cols:
            col_defs.append({
                'name':       c['NAME'],
                'data_type':  c.get('DATA_TYPE', 'ASCII_STRING'),
                'start_byte': int(c['START_BYTE']),
                'bytes':      int(c['BYTES'])
            })
        out.append({
            'pointer':      ptr,
            'record_bytes': int(rec_bytes),
            'file_records': int(row_cnt),
            'columns':      col_defs
        })
    return out
def load_pds3_tab(lbl_path, tab_path=None, table_index=0):
    """
    Load one of the tables from a PDS3 .LBL + .TAB pair.

    Parameters
    ----------
    lbl_path     : str
      Path to the .LBL file.
    tab_path     : str, optional
      If provided, uses this .TAB file; otherwise uses the '^TABLE' entry.
    table_index  : int
      Which TABLE to pick (0-based).

    Returns
    -------
    data : dict of numpy.ndarray
      Keys are column names, values are 1D arrays of length=file_records.
    """
    tables = parse_pds3_label(lbl_path)
    try:
        meta = tables[table_index]
    except IndexError:
        raise IndexError(f"Label only contains {len(tables)} table(s); index {table_index} is out of range.")
    if tab_path is None:
        if meta['pointer'] is None:
            raise ValueError("No '^TABLE' filename in label for table index "
                             f"{table_index}; please pass tab_path explicitly.")
        tab_path = meta['pointer']
    nrows = meta['file_records']
    data  = {}
    for col in meta['columns']:
        name   = col['name']
        length = col['bytes']
        dt     = col['data_type']
        if dt == 'ASCII_REAL':
            data[name] = np.empty(nrows, dtype=float)
        elif dt == 'ASCII_INTEGER':
            data[name] = np.empty(nrows, dtype=int)
        else:
            data[name] = np.empty(nrows, dtype=f'U{length}')
    with open(tab_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= nrows:
                break
            for col in meta['columns']:
                name  = col['name']
                start = col['start_byte'] - 1
                length= col['bytes']
                raw   = line[start:start+length].strip()
                arr   = data[name]
                if arr.dtype == float:
                    arr[i] = float(raw) if raw else np.nan
                elif arr.dtype == int:
                    arr[i] = int(raw)   if raw else 0
                else:
                    arr[i] = raw
    return data
def load_all_from_lbl(base_folder, file_name_unc, return_flag_mask=True, cal=False):
    """
    Load cube, metadata, flag mask, and housekeeping data using just the base folder and LBL filename prefix.
    
    Returns:
        data       : np.ndarray     - the calibrated or raw image cube
        meta       : dict           - metadata parsed from .LBL
        flag_mask  : np.ndarray     - mask showing null/saturated values
        hk_data    : dict           - extracted housekeeping telemetry
    """
    lbl_file = os.path.join(base_folder, file_name_unc + "_1.LBL")
    qub_file = os.path.join(base_folder, file_name_unc + "_1.QUB")
    lbl_hk_file = os.path.join(base_folder, file_name_unc + "_HK_1.LBL")
    tab_hk_file = os.path.join(base_folder, file_name_unc + "_HK_1.TAB")
    data, meta, flag_mask = load_qub_from_lbl(lbl_file, qub_file, cal=cal, return_flag_mask=return_flag_mask)
    hk_data = extract_hk_data(lbl_hk_file, tab_hk_file)
    return data, meta, flag_mask, hk_data


def find_calibration_lbl(base_folder):
    calibration_basenames = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if "_HK" in file.upper():
                continue
            if file.lower().endswith(".lbl"):
                lbl_path = os.path.join(root, file)
                try:
                    metadata = parse_lbl_metadata(lbl_path)
                    target_type = metadata.get("target_type", "").upper()
                    exposure_time = metadata.get("exposure_time", None)
                    if target_type == "CALIBRATION" and (exposure_time == 0 or exposure_time is None):
                        base_name = os.path.splitext(file)[0]
                        for ext in [".QUB", ".CUB"]:
                            qub_path = os.path.join(root, base_name + ext)
                            if os.path.exists(qub_path):
                                base_id = "_".join(base_name.split("_")[:-1])  # Remove trailing _1
                                calibration_basenames.append(base_id)
                                print(f"Found CALIBRATION with exposure_time = 0: {base_id}")
                                break
                        else:
                            print(f"CALIBRATION label without matching QUB/CUB: {lbl_path}")
                    else:
                        if exposure_time != 0 and exposure_time is not None:
                            print(f"Skipping label with exposure_time != 0: {lbl_path}")
                except Exception as e:
                    print(f"Failed to parse {lbl_path}: {e}")
    return calibration_basenames

def find_dark_qub(base_folder):
    dark_basenames = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if "_HK" in file.upper():
                continue
            if file.lower().endswith(".lbl"):
                lbl_path = os.path.join(root, file)
                try:
                    metadata = parse_lbl_metadata(lbl_path)
                    target_type = metadata.get("target_type", "").upper()
                    target_name = metadata.get("target_name", "").upper()
                    if target_type == "CALIBRATION" and target_name == "DARK":
                        base_name = os.path.splitext(file)[0]
                        for ext in [".QUB", ".CUB"]:
                            qub_path = os.path.join(root, base_name + ext)
                            if os.path.exists(qub_path):
                                base_id = "_".join(base_name.split("_")[:-1])
                                dark_basenames.append(base_id)
                                print(f"Found DARK calibration: {base_id}")
                                break
                        else:
                            print(f"DARK calibration label without matching QUB/CUB: {lbl_path}")
                except Exception as e:
                    print(f"Failed to parse {lbl_path}: {e}")
    return dark_basenames