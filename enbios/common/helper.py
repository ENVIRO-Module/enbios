import json
from typing import List

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.writer.excel import save_virtual_workbook

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


def generate_workbook(cmds):
    # Convert list of pd.DataFrames to Excel workbook
    wb = Workbook(write_only=True)
    ws_count = 0
    for name, df in cmds:
        if df.shape[0] < 2:
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
