"""
A script to read a list of .spold files and obtain a NIS file with the following commands

* InterfaceTypes
* BareProcessors
* Interfaces

"""
import logging
import operator
import os
import traceback

import math
import xlrd
from nexinfosys.command_generators.parser_ast_evaluators import get_nis_name
from nexinfosys.common.helper import download_file
from io import StringIO
import pandas as pd
from bw2io.importers.ecospold2 import Ecospold2DataExtractor
from enbios.common.helper import list_to_dataframe, generate_json, generate_workbook


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
    if os.path.exists(f) and os.path.isfile(f):
        return Ecospold2DataExtractor.extract(f, "test", use_mp=False)
    else:
        return None


def lci_to_interfaces_csv(lci_file, csv_file):
    lci_tmp = read_ecospold_file(lci_file)
    s = generate_csv(lci_tmp[0]["exchanges"])
    with open(csv_file, "wt") as f:
        f.write(s)

"""
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/Electricity, PV, 3kWp, multi-Si.spold")
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/Electricity, PV, production mix.spold")
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/5c5c2277-8df1-4a73-a479-9c14deec9bb1.spold")
s = generate_json(lci_tmp[0]["exchanges"])
with open("/home/rnebot/Downloads/exchanges.json.txt", "wt") as f:
    f.write(s)
s = generate_csv(lci_tmp[0]["exchanges"])
with open("/home/rnebot/Downloads/exchanges.csv", "wt") as f:
    f.write(s)

"""


class SpoldToNIS:
    def __init__(self):
        pass

    def _get_spold_files_from_correspondence_file(self, correspondence_path):
        df = pd.read_csv(correspondence_path, comment="#")
        # Assume all columns are present: "name", "match_target_type", "match_target", "weight", "match_conditions"
        spolds = []
        ecoinvent_filename_idx = None
        type_idx = None
        name_idx = None
        for idx, col in enumerate(df.columns):
            if col.lower() == "match_target":
                ecoinvent_filename_idx = idx
            elif col.lower() == "match_target_type":
                type_idx = idx
            elif col.lower() == "name":
                name_idx = idx
        if ecoinvent_filename_idx is not None and type_idx is not None and name_idx is not None:
            for idx, r in df.iterrows():
                ecoinvent_filename = r[df.columns[ecoinvent_filename_idx]]
                o_type = r[df.columns[type_idx]]
                name = r[df.columns[name_idx]]
                if ecoinvent_filename != "" and o_type != "" and name != "":
                    if o_type.lower() in ["lca", "lci"]:
                        spolds.append(dict(name=name, ecoinvent_filename=ecoinvent_filename))
        return spolds

    def _get_spold_files_from_nis_file(self, nis_url):
        bytes_io = download_file(nis_url)
        xl = pd.ExcelFile(bytes_io.getvalue(), engine='openpyxl')
        # xl = pd.ExcelFile(xlrd.open_workbook(file_contents=bytes_io.getvalue()), engine="xlrd")
        spolds = []
        for sheet_name in xl.sheet_names:
            if not sheet_name.lower().startswith("bareprocessors"):
                continue
            df = xl.parse(sheet_name, header=0)
            ecoinvent_filename_idx = None
            ecoinvent_name_idx = None
            name_idx = None
            for idx, col in enumerate(df.columns):
                if col.lower() == "@ecoinventfilename":
                    ecoinvent_filename_idx = idx
                elif col.lower() == "@ecoinventname":
                    ecoinvent_name_idx = idx
                elif col.lower() == "processor":
                    name_idx = idx
            if ecoinvent_filename_idx is not None and ecoinvent_name_idx is not None and name_idx is not None:
                for idx, r in df.iterrows():
                    ecoinvent_filename = r[df.columns[ecoinvent_filename_idx]]
                    ecoinvent_name = r[df.columns[ecoinvent_name_idx]]
                    name = r[df.columns[name_idx]]
                    if ecoinvent_filename != "" and not isinstance(ecoinvent_filename, float) and ecoinvent_name != "" and name != "":
                        spolds.append(dict(name=name, ecoinvent_filename=ecoinvent_filename, ecoinvent_name=ecoinvent_name))
        return spolds

    def spold2nis(self, default_output_interface: str, lci_base: str, correspondence, nis_base_url: str, output_file: str):
        """
        A method to transform Spold files into a Workbook with InterfacesTypes, BareProcessors and Interfaces

        :param default_output_interface: Name of the interface to use when no clear output interface is found in the .spold file
        :param lci_base: Local path where .spold files are located
        :param correspondence: Location of the correspondence file (open "sim_correspondence_example.csv" file)
        :param nis_base_url: Location of the NIS Base file, which can also have processors with LCI location
        :param nis_file: Path of the result workbook
        :return: None
        """

        # Read the list of Spold files to process
        spolds = []
        if correspondence:
            _ = self._get_spold_files_from_correspondence_file(correspondence)
            spolds.extend(_)
        if nis_base_url:
            _ = self._get_spold_files_from_nis_file(nis_base_url)
            spolds.extend(_)

        interface_types = {}
        processors = {}
        interfaces = {}

        # Process each .spold file
        for spold in spolds:
            file_name = f"{lci_base}{os.sep}{spold['ecoinvent_filename']}"
            try:
                lci = read_ecospold_file(file_name)
                if lci is None:
                    msg = f"Ecospold file '{file_name}' not found"
            except:
                lci = None
                msg = f"Exception reading file '{file_name}'."
                traceback.print_exc()

            if lci is None:
                logging.debug(msg)
                continue
            lci = lci[0]
            # Obtain processor name
            p_name = lci["name"]
            # Processors
            processors[get_nis_name(p_name)] = dict(name=p_name, lca_code=lci["activity"],
                                                    lca_type=lci["activity type"], lca_file=lci["filename"],
                                                    description=lci["comment"])

            # Read exchanges (to produce InterfaceTypes and Interfaces)
            try:
                j = generate_json(lci["exchanges"])
                df = pd.read_json(StringIO(j))
            except:
                traceback.print_exc()
                logging.debug(f"Problem converting '{file_name}' 'exchanges' to JSON. Skipped.")
                continue

            del df["activity"]
            del df["classifications"]
            del df["properties"]
            del df["loc"]
            del df["production volume"]
            del df["uncertainty type"]
            dfi = df.copy()

            # Interface Types
            del dfi["amount"]
            # There are duplicate "name"s, for now remove them
            dfi.drop_duplicates(subset=["name"], inplace=True)
            for idx, r in dfi.iterrows():
                sphere = "Technosphere" if r["type"].lower() != "biosphere" else "Biosphere"
                interface_types[get_nis_name(r["name"])] = dict(comment=r.get("comment", ""), flow=r["flow"], sphere=sphere, unit=r["unit"], lci_name=r["name"])
            # Interfaces
            names = {k: v for k, v in zip(df['name'].str.lower().values, df['name'].values)}
            tmp = df.groupby([df['name'].str.lower()]).sum()  # Acumulate (sum) repeated exchange names
            # Find main output
            main_output = None
            main_output_is_output = True
            for idx, r in tmp.iterrows():
                if r["amount"] == 1.0:
                    main_output = get_nis_name(names[idx.lower()])
                    break
            if main_output is None:
                # Special activities can be scaled according to a -1 exchange (Nick clarified this)
                for idx, r in tmp.iterrows():
                    if r["amount"] == -1.0:
                        main_output = get_nis_name(names[idx.lower()])
                        main_output_is_output = False
                        break
            if main_output is None:
                main_output = default_output_interface
            # Add interfaces, output interface first
            i_name = get_nis_name(names.get(main_output, main_output))
            interfaces[(p_name, i_name)] = dict(value=1, relative_to="", is_output=main_output_is_output)
            for idx, r in tmp.iterrows():
                i_name = get_nis_name(names[idx])
                if (p_name, i_name) not in interfaces:
                    relative_to = main_output if i_name != main_output else ""
                    # value = r["amount"] if relative_to != "" else ""
                    interfaces[(p_name, i_name)] = dict(value=r["amount"], relative_to=relative_to,
                                                        is_output=i_name == main_output)

        # Generate the three commands, InterfaceTypes, BareProcessors, Interfaces
        cmds = []

        # InterfaceTypes
        header = ["InterfaceTypeHierarchy", "InterfaceType", "Sphere", "RoegenType", "ParentInterfaceType", "Formula",
                  "Description", "Unit", "OppositeSubsystemType", "Attributes", "@EcoinventName"]
        lst = []
        if default_output_interface != "":
            lst.append(["LCI", default_output_interface, "Technosphere", "Flow", "", "",
                        "Default-generic output for LCI activities which do not state explicitly its output interface",
                        "EJ", "", "", ""])
        for interface_type, props in interface_types.items():
            opposite = "Environment" if props["sphere"].lower() == "biosphere" else ""
            lst.append(["LCI", interface_type, props["sphere"], "Flow", "", "",
                        props["comment"], props["unit"], opposite, "", props["lci_name"]])
        lst = sorted(lst, key=operator.itemgetter(1))
        lst.insert(0, header)
        cmds.append(("InterfaceTypes", list_to_dataframe(lst)))

        # BareProcessors
        lst = [["ProcessorGroup", "Processor", "ParentProcessor", "SubsystemType", "System", "FunctionalOrStructural",
                "Accounted", "Stock", "Description", "GeolocationRef", "GeolocationCode", "GeolocationLatLong",
                "Attributes", "@EcoinventName", "@EcoinventFilename", "@region"]]
        for processor, props in processors.items():
            spold_file = props["lca_file"]
            lst.append(["EcoinventReferenceStructurals", processor, "", "", "", "Structural", "No", "", "", "", "", "", "", props["name"], spold_file, ""])
        cmds.append(("BareProcessors", list_to_dataframe(lst)))

        # Interfaces
        lst = [
            ["Processor", "InterfaceType", "Interface", "Sphere", "RoegenType", "Orientation", "OppositeSubsystemType",
             "GeolocationRef", "GeolocationCode", "InterfaceAttributes", "Value", "Unit", "RelativeTo", "Uncertainty",
             "Assessment", "PedigreeMatrix", "Pedigree", "Time", "Source", "NumberAttributes", "Comments"]]
        for p_i, props in interfaces.items():
            p_name = get_nis_name(p_i[0])
            v = props["value"]
            orientation = "Output" if props["is_output"] else "Input"
            output_name = props["relative_to"]
            lst.append([p_name, p_i[1], "", "", "", orientation, "", "", "", "", v,
                        "", output_name, "", "", "", "", "Year", "Ecoinvent", "", ""])
        cmds.append(("Interfaces", list_to_dataframe(lst)))

        s = generate_workbook(cmds)
        if s:
            with open(output_file, "wb") as f:
                f.write(s)
        else:
            print(f"ACHTUNG BITTE!: it was not possible to produce XLSX, probably because no .Spold file was found, "
                  f"check LCI data folder, {lci_base}, is correctly specified.")
