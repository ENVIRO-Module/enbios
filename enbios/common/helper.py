import json

import numpy as np

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
