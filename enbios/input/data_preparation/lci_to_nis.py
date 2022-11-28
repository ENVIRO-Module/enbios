"""
A script to read a list of .spold files and obtain a NIS file with the following commands

* InterfaceTypes
* BareProcessors
* Interfaces

"""
import logging
import operator
import os
import sys
import traceback
from os import listdir
from os.path import isfile, join

from nexinfosys.bin.cli_script import print_issues, PrintColors
from nexinfosys.command_generators import Issue, IType
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


class MyEcoSpold2DataExtractor(Ecospold2DataExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def extract_properties(cls, exc):
        properties = {}

        for obj in exc.iterchildren():
            if obj.tag.endswith("property"):
                properties[obj.name.text] = {"amount": float(obj.get("amount"))}
                if hasattr(obj, "unitName"):
                    properties[obj.name.text]["unit"] = obj.unitName.text
                if hasattr(obj, "comment"):
                    properties[obj.name.text]["comment"] = obj.comment.text
            elif obj.tag.endswith("compartment"):
                properties["compartment"] = obj.compartment.text
                properties["subcompartment"] = obj.subcompartment.text

        return properties


def read_ecospold_file(f):
    lci = None
    issues = []
    found_name = None
    for name in [f, f"{f}.spold"]:
        try:
            if os.path.exists(name) and os.path.isfile(name):
                found_name = name
                _original_stdout = sys.stdout
                _original_stderr = sys.stderr
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = sys.stdout
                lci = MyEcoSpold2DataExtractor.extract(name, "test", use_mp=False)
                sys.stdout.close()
                sys.stdout = _original_stdout
                sys.stderr = _original_stderr
                lci = lci[0]
                break
        except:
            issues.append(Issue(itype=IType.ERROR, description=f"Exception reading file '{f}': {traceback.format_exc()}"))
            # traceback.print_exc()

    if not found_name:
        issues.append(Issue(itype=IType.ERROR, description=f"File '{f}' (or {f}.spold) not found. Please check '@EcoinventFilename' column in the NIS base file or if the file has not been downloaded into the LCI files folder"))

    return lci, found_name, issues


def lci_to_interfaces_csv(lci_file, csv_file):
    lci_tmp, found_name, issues = read_ecospold_file(lci_file)
    s = generate_csv(lci_tmp["exchanges"])
    with open(csv_file, "wt") as f:
        f.write(s)

"""
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/Electricity, PV, 3kWp, multi-Si.spold")
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/Electricity, PV, production mix.spold")
lci_tmp = read_ecospold_file("/home/rnebot/Downloads/5c5c2277-8df1-4a73-a479-9c14deec9bb1.spold")
s = generate_json(lci_tmp["exchanges"])
with open("/home/rnebot/Downloads/exchanges.json.txt", "wt") as f:
    f.write(s)
s = generate_csv(lci_tmp[0]["exchanges"])
with open("/home/rnebot/Downloads/exchanges.csv", "wt") as f:
    f.write(s)

"""


class SpoldToNIS:
    def __init__(self):
        pass

    @staticmethod
    def _get_spold_files_from_correspondence_file(correspondence_path):  # TODO UNUSED
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

    @staticmethod
    def _get_spold_files_from_nis_file(nis_url):
        """
        Elaborate a list of Spold files (to be later processed) coming from a NIS formatted XLSX file (nis_url).

        The file is examined, looking for columns (row 1) named "processor", "@ecoinventfilename" and "@ecoinventname"
        in "BareProcessors" sheets to construct each of the entries (each entry is a "dict") of the returned list.

        :param nis_url: Location of NIS formatted XLSX
        :return: List of dict's, where each dict contains the information of a Spold file
        """
        bytes_io = download_file(nis_url)
        xl = pd.ExcelFile(bytes_io.getvalue(), engine='openpyxl')
        # xl = pd.ExcelFile(xlrd.open_workbook(file_contents=bytes_io.getvalue()), engine="xlrd")
        spolds = []
        for sheet_name in xl.sheet_names:
            if not sheet_name.lower().startswith("bareprocessors"):
                continue
            df = xl.parse(sheet_name, header=0)
            ecoinvent_filename_idx = None
            ecoinvent_carrier_name_idx = None
            name_idx = None
            for idx, col in enumerate(df.columns):
                if col.lower() == "@ecoinventfilename":
                    ecoinvent_filename_idx = idx
                elif col.lower() == "@ecoinventcarriername":
                    ecoinvent_carrier_name_idx = idx
                elif col.lower() == "processor":
                    name_idx = idx
            if ecoinvent_filename_idx is not None and name_idx is not None:
                for idx, r in df.iterrows():
                    ecoinvent_filename = r[df.columns[ecoinvent_filename_idx]]
                    if ecoinvent_carrier_name_idx is not None:
                        ecoinvent_carrier_name = r[df.columns[ecoinvent_carrier_name_idx]]
                    else:
                        ecoinvent_carrier_name = None
                    name = r[df.columns[name_idx]]
                    if ecoinvent_filename != "" and not isinstance(ecoinvent_filename, float) and name != "":
                        spolds.append(dict(name=name,
                                           ecoinvent_filename=ecoinvent_filename,
                                           ecoinvent_carrier_name=ecoinvent_carrier_name))
        return spolds

    def spold2nis(self, default_output_interface: str, lci_base: str, correspondence, nis_base_url: str, output_file: str):
        """
        A method to transform Spold files into a NIS Workbook with InterfacesTypes, BareProcessors and Interfaces

        :param default_output_interface: Name of the interface to use when no clear output interface is found in the .spold file
        :param lci_base: Local path where .spold files are located
        :param correspondence: Location of the correspondence file (open "sim_correspondence_example.csv" file)
        :param nis_base_url: Location of the NIS Base file, which can also have processors with LCI location
        :param output_file: Path of the resulting Workbook
        :return: None
        """

        # Read exchanges into a pd.DataFrame (to produce InterfaceTypes and Interfaces)
        def _lci_to_dataframe(lci_in):
            try:
                j = generate_json(lci_in["exchanges"])
                df_tmp = pd.read_json(StringIO(j))
            except:
                traceback.print_exc()
                logging.debug(f"Problem converting '{file_name}' 'exchanges' to JSON. Skipped.")
                return None, None

            for c in ["activity", "classifications", "properties", "loc", "production volume", "uncertainty type"]:
                del df_tmp[c]
            dfi_tmp = df_tmp.copy()
            del dfi_tmp["amount"]
            # There are duplicate "name"s, for now remove them
            dfi_tmp.drop_duplicates(subset=["name"], inplace=True)
            return df_tmp, dfi_tmp

        def _find_main_output(aggregate_exchanges):
            """ Find main output """
            _main_output = None
            _main_output_is_output = None
            for _idx, _r in aggregate_exchanges.iterrows():
                if _r["amount"] == 1.0:
                    _main_output = get_nis_name(_r["name"])
                    _main_output_is_output = True
                    break
            if _main_output is None:
                # Special activities can be scaled according to a -1 exchange
                # (Nick clarified this unusual but possible case)
                for _idx, _r in aggregate_exchanges.iterrows():
                    if _r["amount"] == -1.0:
                        _main_output = get_nis_name(_r["name"])
                        _main_output_is_output = False
                        break
            if _main_output is None:
                _main_output = default_output_interface
                _main_output_is_output = True

            return _main_output, _main_output_is_output

        def _generate_commands(iface_types, procs, ifaces):
            """ Generate the three commands, InterfaceTypes, BareProcessors, Interfaces """
            _cmds = []

            # InterfaceTypes
            header = ["InterfaceTypeHierarchy", "InterfaceType", "Sphere", "RoegenType", "ParentInterfaceType",
                      "Formula",
                      "Description", "Unit", "OppositeSubsystemType", "Attributes", "@EcoinventName"]
            lst = []
            if default_output_interface != "":
                lst.append(["LCI", default_output_interface, "Technosphere", "Flow", "", "",
                            "Default-generic output for LCI activities not stating explicitly its output interface",
                            "EJ", "", "", ""])
            for interface_type, props in iface_types.items():
                opposite = "Environment" if props["sphere"].lower() == "biosphere" else ""
                lst.append(["LCI", interface_type, props["sphere"], "Flow", "", "",
                            props["comment"], props["unit"], opposite, "", props["lci_name"]])
            lst = sorted(lst, key=operator.itemgetter(1))
            lst.insert(0, header)
            _cmds.append(("InterfaceTypes", list_to_dataframe(lst)))

            # BareProcessors
            lst = [
                ["ProcessorGroup", "Processor", "ParentProcessor", "SubsystemType", "System", "FunctionalOrStructural",
                 "Accounted", "Stock", "Description", "GeolocationRef", "GeolocationCode", "GeolocationLatLong",
                 "Attributes", "@EcoinventName", "@EcoinventFilename", "@EcoinventCarrierName", "@region"]]
            for processor, props in procs.items():
                spold_file = props["lca_file"]
                lst.append(
                    ["EcoinventReferenceStructurals", processor, "", "", "", "Structural", "No", "", "", "", "", "",
                     "", props["name"], spold_file, props["carrier_name"], ""])
            _cmds.append(("BareProcessors", list_to_dataframe(lst)))

            # Interfaces
            lst = [
                ["Processor", "InterfaceType", "Interface", "Sphere", "RoegenType", "Orientation",
                 "OppositeSubsystemType",
                 "GeolocationRef", "GeolocationCode", "I@compartment", "I@subcompartment",
                 "Value", "Unit", "RelativeTo",
                 "Uncertainty",
                 "Assessment", "PedigreeMatrix", "Pedigree", "Time", "Source", "NumberAttributes", "Comments"]]
            for p_i, props in ifaces.items():
                v = props["value"]
                orientation = "Output" if props["is_output"] else "Input"
                output_name = props["relative_to"]
                iface = p_i[2] if len(p_i) > 2 else ""
                comp = props["compartment"] if "compartment" in props else ""
                subcomp = props["subcompartment"] if "subcompartment" in props else ""
                lst.append([p_i[0], p_i[1], iface, "", "", orientation,
                            "",
                            "", "", comp, subcomp,
                            v, "", output_name,
                            "",
                            "", "", "", "Year", "Ecoinvent", "", ""])
            _cmds.append(("Interfaces", list_to_dataframe(lst)))
            return _cmds

        # FUNCTION STARTS HERE -----------------
        issues = []

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
        files_in_lci_base = set([join(lci_base, f) for f in listdir(lci_base) if isfile(join(lci_base, f))])
        already_processed = set()
        for spold in spolds:
            file_name = os.path.join(f"{lci_base}", f"{spold['ecoinvent_filename']}")
            if file_name in already_processed:
                continue
            else:
                already_processed.add(file_name)

            # Read Spold file
            lci, found_name, _ = read_ecospold_file(file_name)
            issues.extend(_)
            if lci is None:
                continue
            if found_name:
                issues.append(Issue(itype=IType.INFO, description=f"LCI file '{os.path.basename(found_name)}' read correctly"))
                files_in_lci_base.remove(found_name)

            # Bring compartment and subcompartment to exchange properties level
            for exc in lci["exchanges"]:
                exc["compartment"] = exc["properties"].get("compartment")
                exc["subcompartment"] = exc["properties"].get("subcompartment")

            # Transform Spold file into usable data structures (pd.DataFrame's)
            df, dfi = _lci_to_dataframe(lci)
            if df is None:
                continue

            # Obtain processor name
            p_name = lci["name"]

            # PROCESSORS
            nis_name = get_nis_name(p_name)
            if nis_name in processors:
                cont = 1
                while f"{nis_name}_{cont}" in processors:
                    cont += 1
                nis_name = f"{nis_name}_{cont}"
            processors[nis_name] = dict(name=p_name, lca_code=lci["activity"],
                                        lca_type=lci["activity type"], lca_file=lci["filename"],
                                        carrier_name=spold.get("ecoinvent_carrier_name", ""),
                                        description=lci["comment"])

            # INTERFACE TYPES <- dfi
            for idx, r in dfi.iterrows():
                sphere = "Technosphere" if r["type"].lower() != "biosphere" else "Biosphere"
                interface_types[get_nis_name(r["name"])] = dict(comment=r.get("comment", ""),
                                                                flow=r["flow"],
                                                                sphere=sphere,
                                                                unit=r["unit"],
                                                                lci_name=r["name"])

            # INTERFACES <- df

            # Find main output
            main_output, main_output_is_output = _find_main_output(df)

            # Add output interface first
            i_name = get_nis_name(main_output)
            interfaces[(nis_name, i_name)] = dict(value=1, relative_to="", is_output=main_output_is_output)
            # Add rest of interfaces
            for idx, r in df.iterrows():
                compartment = r["compartment"]
                subcompartment = r["subcompartment"]
                i_name = get_nis_name(r["name"])
                it_name = i_name
                if compartment:
                    i_name += f"_{get_nis_name(compartment)}"
                if subcompartment:
                    i_name += f"_{get_nis_name(subcompartment)}"
                if (nis_name, i_name) not in interfaces:
                    relative_to = main_output if i_name != main_output else ""
                    # value = r["amount"] if relative_to != "" else ""
                    interfaces[(nis_name, it_name, i_name)] = dict(value=r["amount"],
                                                                   relative_to=relative_to,
                                                                   compartment=compartment,
                                                                   subcompartment=subcompartment,
                                                                   is_output=i_name == main_output)
        for f in files_in_lci_base:
            issues.append(Issue(itype=IType.WARNING, description=f"File '{os.path.basename(f)}' not used"))

        # Generate InterfaceTypes, BareProcessors and Interfaces commands
        cmds = _generate_commands(interface_types, processors, interfaces)

        # Generate an in-memory XLSX file
        s = generate_workbook(cmds)

        # Write to file
        if s:
            with open(output_file, "wb") as f:
                f.write(s)

        # Print collected issues
        print_issues("Spold2NIS", "", issues, f"Please check the issues and output file '{output_file}'")

        if s:
            print(f"\nNIS file:\n"
                  f"{PrintColors.BOLD}{output_file}{PrintColors.END}\n"
                  f"containing information from LCI files generated successfully.\n"
                  f"Please, refer to this file from the NIS Base file, using an ImportCommands command (worksheet).\n"
                  f"If NIS Base file is a Google Sheets document:\n"
                  f" * Upload the generated file to Google Drive,\n"
                  f" * Go to the uploaded file in Google Drive, open right click menu, 'Get Link' (select 'anybody with the link') and\n"
                  f" * Copy/paste the link into the appropriate 'ImportCommands' cell (under 'Workbook' column) of NIS base file.")
