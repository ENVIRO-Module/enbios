from typing import Tuple

import pandas as pd

from enbios.common.helper import isfloat
from enbios.model import SimStructuralProcessorAttributes


def create_register_or_update_processor_and_attributes(prd,
                                                       tech, region, carrier, scenario, time_, subtech, subscenario,
                                                       carrier_idx,
                                                       attrs):
    # tech, region, carrier, scenario, time_, subtech, subscenario -> for col in df.columns
    k = SimStructuralProcessorAttributes.partial_key(tech, region, carrier, scenario, time_,
                                                     subtech, subscenario)
    if carrier_idx >= 0 and not carrier:
        return None
    if tech is None or not region or region == "-":
        # print(f"WARNING: Missing Tech and/or Region: {k}. Skipped")
        return None
    # (tech, region, scenario, time) - (cols)
    o = prd.get(k)
    if len(o) == 0:
        pa = SimStructuralProcessorAttributes(tech, region, carrier, scenario, time_, subtech, subscenario)
        prd.put(k, pa)
    elif len(o) == 1:
        pa = o[0]
    else:
        raise Exception(f"Found {len(o)} occurrences of SimStructuralProcessorAttributes: {k}")
    # Variables from current pd.DataFrame
    _ = pa.attrs
    for key, v in attrs.items():
        if isfloat(v):
            v = float(v)
        if key not in _:
            _[key] = v
        else:
            if isinstance(v, float):
                tmp = _[key]
                _[key] += v  # SUM
                if v != 0 and tmp == _[key]:
                    raise Exception(f"ERROR: did not change. value: {v}, previous: {tmp}")
            else:
                _[key] = v  # Overwrite
    return pa


def find_column_idx_name(columns: pd.Index, possible_values: list) -> Tuple[int, str]:
    if any(columns.isin(possible_values)):
        for p in possible_values:
            try:
                return columns.get_loc(p), p
            except:
                continue
    else:
        return -1, None
