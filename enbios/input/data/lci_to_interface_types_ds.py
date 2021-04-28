from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from bw2io.importers.ecospold2 import Ecospold2DataExtractor

from enbios.common.helper import generate_json


def generate_csv(o):
    js = generate_json(o)
    df = pd.read_json(js)
    del df["activity"]
    del df["classifications"]
    del df["properties"]
    del df["loc"]
    del df["production volume"]
    del df["uncertainty type"]
    del df["amount"]  # We are generating a reference of Interface Types. Not specific to a particular Activity
    # TODO There are many duplicate "name"s, for now remove them
    df.drop_duplicates(subset=["name"], inplace=True)
    s = StringIO()
    s.write('# To regenerate this file, execute function "generate_csv" in script "read_lci_data (ENVIRO)"\n')
    df.to_csv(s, index=False)
    return s.getvalue()


def read_ecospold_file(f):
    return Ecospold2DataExtractor.extract(f, "test", use_mp=False)


lci_tmp = read_ecospold_file("/home/rnebot/Downloads/Electricity, PV, 3kWp, multi-Si.spold")
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/Electricity, PV, production mix.spold")
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/5c5c2277-8df1-4a73-a479-9c14deec9bb1.spold")
s = generate_json(lci_tmp[0]["exchanges"])
with open("/home/rnebot/Downloads/exchanges.json.txt", "wt") as f:
    f.write(s)
s = generate_csv(lci_tmp[0]["exchanges"])
with open("/home/rnebot/Downloads/exchanges.csv", "wt") as f:
    f.write(s)
