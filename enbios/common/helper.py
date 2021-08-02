import copy
import json
import os
import re
import tempfile
import urllib.request
from io import BytesIO
from typing import List, Union, BinaryIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from nexinfosys.command_generators import IType
from nexinfosys.common.helper import download_file
from nexinfosys.embedded_nis import NIS
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.writer.excel import save_virtual_workbook

from enbios.processing import read_submit_solve_nis_file

JSON_INDENT = 4
ENSURE_ASCII = False


def _json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    from datetime import datetime

    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial
    elif isinstance(obj, np.int64):
        return int(obj)
    elif getattr(obj, "Schema"):
        # Use "marshmallow"
        return getattr(obj, "Schema")().dump(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def generate_json(o):
    return json.dumps(o,
                      default=_json_serial,
                      sort_keys=True,
                      indent=JSON_INDENT,
                      ensure_ascii=ENSURE_ASCII,
                      separators=(',', ': ')
                      ) if o else None


def list_to_dataframe(lst: List) -> pd.DataFrame:
    return pd.DataFrame(data=lst[1:], columns=lst[0])


def get_scenario_name(prefix, s):
    def is_int(si):
        try:
            int(si)
            return True
        except ValueError:
            return False

    return f"{prefix}{s}" if is_int(s) else str(s)


def generate_workbook(cmds, generate_empty=False):
    # Convert list of pd.DataFrames to Excel workbook
    wb = Workbook(write_only=True)
    ws_count = 0
    for name, df in cmds:
        if df.shape[0] == 0:
            if not generate_empty:
                continue

        ws_count += 1
        ws = wb.create_sheet(name)
        widths = [0]*(df.shape[1]+1)  # A maximum of 100 columns
        max_columns = 0
        for r in dataframe_to_rows(df, index=False, header=True):
            if len(r) > max_columns:
                max_columns = len(r)
            for i in range(len(r)):
                width = int(len(str(r[i])) * 1.1)
                if width > widths[i]:
                    widths[i] = width

        for i, column_width in enumerate(widths):
            ws.column_dimensions[get_column_letter(i+1)].width = column_width

        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)

    if ws_count > 0:
        return save_virtual_workbook(wb)
    else:
        return None


def hash_array(f: Union[str, bytes]):
    import hashlib
    m = hashlib.md5()
    if isinstance(f, str):
        m.update(f.encode("utf-8"))
    else:
        m.update(f)
    return m.digest()


def set_zip_timestamp(in_zip: Union[os.PathLike, str, BinaryIO], timestamp=(2020, 1, 1, 0, 0, 0)) -> BinaryIO:
    """
    Modify the timestamp of all files, to stabilize the hash

    :param in_zip: Zip file whose files timestamp will be modified
    :param timestamp: Tuple with the timestamp to set for all the files
    :return A BytesIO with the resulting Zip
    """
    out_zip = BytesIO()
    with ZipFile(in_zip, mode="r") as zin:
        with ZipFile(out_zip, mode="w") as zout:
            for zinfo in zin.infolist():
                data = zin.read(zinfo.filename)
                zinfo_new = copy.copy(zinfo)
                zinfo_new.date_time = timestamp
                zout.writestr(zinfo_new, data)
    return out_zip


def get_valid_name(original_name):
    prefix = original_name[0] if original_name[0].isalpha() else "_"
    remainder = original_name[1:] if original_name[0].isalpha() else original_name
    return prefix + re.sub("[^0-9a-zA-Z_]+", "", remainder)


def get_file_url(fn):
    return "file:" + urllib.request.pathname2url(fn)


def prepare_base_state(base_url: str, solve: bool, directory: str = None):
    from nexinfosys import initialize_configuration
    initialize_configuration()  # Needed to make download and NIS later work properly
    tmp_io = download_file(base_url)
    bytes_io = set_zip_timestamp(tmp_io)
    hash = hash_array(bytes_io.getvalue())
    val_name = get_valid_name(base_url)
    if directory is None:
        directory = tempfile.gettempdir()
    hash_file = os.path.join(f"{directory}", f"base.hash.{val_name}")
    state_file = os.path.join(f"{directory}", f"base.state.{val_name}")
    if os.path.isfile(hash_file) and os.path.isfile(state_file):
        with open(hash_file, "rb") as f:
            cached_hash = f.read()
        update = cached_hash != hash
    else:
        update = True

    if update:
        temp_name = tempfile.NamedTemporaryFile(dir=directory, delete=False)
        temp_name = temp_name.name
        with open(temp_name, "wb") as f:
            f.write(bytes_io.getvalue())
        nis, issues = read_submit_solve_nis_file(get_file_url(temp_name), None, solve=solve)
        os.remove(temp_name)
        any_error = False
        for issue in issues:
            print(issue)
            if issue.itype == IType.ERROR:
                any_error = True
        # Write if there are no errors. IF there is any error generate an Exception
        if any_error:
            raise Exception(f'There were errors with the NIS base file {base_url}.')
        from nexinfosys.serialization import serialize_state
        serial_state = serialize_state(nis.get_state())
        nis.close_session()
        with open(hash_file, "wb") as f:
            f.write(hash)
        with open(state_file, "wb") as f:
            f.write(serial_state)
    else:
        with open(state_file, "rb") as f:
            serial_state = f.read()

    return serial_state
