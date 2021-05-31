from io import StringIO

import openpyxl
import pandas as pd


def convert_lcia_implementation_to_nis(lcia_implementation_file: str, lcia_file: str):
    def get_horizon(ind):
        return ""

    workbook = openpyxl.load_workbook(lcia_implementation_file, data_only=True)
    cfs = workbook["CFs"]
    units = workbook["units"]
    methods = set()
    categories = set()
    indicators = {}  # And its unit
    horizons = set()
    compartments = {}  # Include subcompartments
    for r in range(1, units.max_row):
        method = units.cell(row=r, column=1).value.strip()
        if "superseded" in method.lower() or "obsolete" in method.lower():
            continue
        if "recipe" not in method.lower() or "midpoint" not in method.lower():
            continue

        category = units.cell(row=r, column=2).value.strip()
        indicator = units.cell(row=r, column=3).value.strip()
        unit = units.cell(row=r, column=4).value.strip()

        methods.add(method)
        categories.add(category)
        indicators[indicator] = unit

    print("METHODS")
    print(methods)
    print("INDICATORS")
    print(indicators.keys())
    print("CATEGORIES")
    print(categories)
    _ = []  # Output
    for r in range(1, cfs.max_row):
        method = cfs.cell(row=r, column=1).value.strip()
        if method not in methods:
            continue
        category = cfs.cell(row=r, column=2).value.strip()
        indicator = cfs.cell(row=r, column=3).value.strip()
        # Guess horizon
        h = get_horizon(indicator)
        name = cfs.cell(row=r, column=4).value.strip()
        compartment = cfs.cell(row=r, column=5).value.strip()
        subcompartment = cfs.cell(row=r, column=6).value.strip()
        v = cfs.cell(row=r, column=7).value
        ind_unit = indicators[indicator]

        _.append((method, indicator, ind_unit, name, "", h, v, compartment, subcompartment))

    df = pd.DataFrame(_, columns=["LCIAMethod", "LCIAIndicator", "LCIAIndicatorUnit",
                                  "Interface", "InterfaceUnit",
                                  "LCIAHorizon", "LCIACoefficient",
                                  "Compartment", "Subcompartment"])

    s = StringIO()
    s.write("# To regenerate this file, execute action 'lcia_implementation_to_nis'\n")
    df.to_csv(s, index=False)
    with open(lcia_file, "wt") as f:
        f.write(s.getvalue())


if __name__ == '__main__':
    convert_lcia_implementation_to_nis("/home/rnebot/GoogleDrive/AA_SENTINEL/LCIA_implementation_3.7.1.xlsx", "/home/rnebot/Downloads/test.csv")
