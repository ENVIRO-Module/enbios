# /usr/bin/env python
import os
from typing import List

import fire
import webbrowser

from nexinfosys.bin.cli_script import set_log_level_from_cli_param, prepare_base_state, print_issues
from nexinfosys.common.helper import PartialRetrievalDictionary

from enbios.common.helper import list_to_dataframe, generate_workbook
from enbios.input.data_preparation.lci_to_nis import SpoldToNIS
from enbios.input.data_preparation.lcia_implementation_to_nis import convert_lcia_implementation_to_nis
from enbios.input.data_preparation.recipe_to_nis import convert_recipe_to_nis
from enbios.input.data_preparation.sentinel_to_nis_prep import sentinel_to_prep_file
from enbios.processing import read_parse_configuration
from enbios.processing.main import Enviro
from enbios.visualize import create_dashboard_app

"""
DEVELOPMENT CLI EXECUTION:
 
python -m enbios.bin.script
  <it will show the automatically generated help>

CLI EXECUTION ("enbios" because it is defined with that name in setup.py):
 
enbios (same as previous)
  <it will show the -automatically generated- help>
  
"""


def list_lcia_indicators(nis_file, output_file):

    def query_lcia_methods(state):
        """
        Return commands informing of the available LCIA indicators

        :return:
        """
        lcia_methods_dict = state.get("_lcia_methods")
        if lcia_methods_dict:
            lcia_methods = PartialRetrievalDictionary()
            from random import random
            for k, v in lcia_methods_dict.items():
                # [0] m=Method,
                # [1] t=caTegory,
                # [2] d=inDicator,
                # [3] h=Horizon,
                # [4] i=Interface,
                # [5] c=Compartment,
                # [6] s=Subcompartment,
                _ = dict(m=k[0], t=k[1], d=k[2], h=k[3], i=k[4], c=k[5], s=k[6])
                # NOTE: a random() is generated just to grant that the tuple is unique
                lcia_methods.put(_, (v[0], v[1], v[2], random()))

            keys = lcia_methods.get(dict(m=None, d=None, h=None, c=None, s=None, t=None), just_key=True)

            # Convert to set of tuples, keeping only the desired elements of dictionaries in each tuple, then sort
            different = sorted(set([(key["m"], key["t"], key["d"], key["c"], key["s"], key["h"]) for key in keys]),
                               key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
            # Prepare pseudo "ScalarIndicators" command
            lst = [["IndicatorsGroup", "Indicator", "Local", "Processors", "Formula", "Unit", "AccountNA", "UnitLabel", "Benchmarks",
                    "Description", "Reference"]]
            for t in different:
                _ = {}
                for i, d in enumerate(["m", "t", "d", "c", "s", "h"]):
                    if t[i] is not None:
                        _[d] = t[i]
                o = lcia_methods.get(_)
                if len(o) > 0:
                    unit = o[0][0]
                else:
                    unit = "dimensionless"

                lst.append(["LCIA",
                            f"{t[0]}_{t[1]}_{t[2]}_{t[3]}_{t[4]}",
                            "Yes",
                            "",  # Processors ("": "all processors")
                            f'LCIAMethod("{t[0]}", "{t[1]}", "{t[2]}", "{t[3]}", "{t[4]}", "{t[5]}")',
                            unit,  # LCIA indicator unit
                            # TODO "Yes" (below) not implemented for LCIAMethod
                            "",  # AccountNA? Yes or "" (for No). "NA" stands for "Not Available"
                            unit,  # UnitLabel
                            "",  # Benchmarks
                            f"Method {t[0]}, indicator {t[2]}, horizon {t[5]}, compartment {t[3]}, "
                            f"subcompartment {t[4]}, category {t[1]}. "
                            f"Generated by enbios 'lcia_list_indicators', using a base NIS file "
                            f"where the LCIA methods are defined.",  # Description
                            ""
                            ])
            return [("ScalarIndicators auto LCIA", list_to_dataframe(lst))]
        else:
            return []

    dir, basename = os.path.split(output_file)
    state, serial_state, issues = prepare_base_state(nis_file, False, dir)
    print_issues("NIS base preparation", nis_file, issues, f"Please check the issues and output file '{output_file}'")
    if state:
        cmds = query_lcia_methods(state)
        s = generate_workbook(cmds, True)
        with open(output_file, "wb") as f:
            f.write(s)


class Enbios:
    @staticmethod
    def recipe_to_csv(recipe_file: str, lcia_file: str, log: str = None):  # TODO DEPRECATED, use "lcia_implementation_to_csv"
        """
        Convert an XLSX file with ReCiPe2016 to a CSV file ready to be declared in a DatasetDef which can be imported
        in a "LCIAMethod" command

        Example:
           enbios recipe_to_csv /home/rnebot/GoogleDrive/AA_SENTINEL/workspace_20210331/ReCiPe2016_CFs_v1.1_20180117.xlsx /home/rnebot/GoogleDrive/AA_SENTINEL/case_studies/library/recipe2016_2.csv


        :param recipe_file: The full path of the input ReCiPe2016 XLSX file
        :param lcia_file: The full path of the output CSV file
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :return:
        """
        set_log_level_from_cli_param(log)
        convert_recipe_to_nis(recipe_file, lcia_file)

    @staticmethod
    def lcia_implementation_to_csv(lcia_implementation_file: str,
                                   lcia_file: str,
                                   method_like: str = "",
                                   method_is: List[str] = [],
                                   include_obsolete: bool = False,
                                   use_nis_name_syntax: bool = True,
                                   log: str = None):
        """
        Convert an XLSX file with LCIA implementation (from Ecoinvent) to a CSV file ready to be declared in a
        DatasetDef which can later be imported in a "LCIAMethod" command

        Example:
           enbios lcia_implementation_to_csv /home/rnebot/GoogleDrive/AA_SENTINEL/LCIA_implementation_3.7.1.xlsx /home/rnebot/GoogleDrive/AA_SENTINEL/case_studies/library/lcia_implementation_nis.csv

           Just one method, using "method-is" ("method-like" is for methods containing the string):
           enbios lcia_implementation_to_csv --method-is "ReCiPe Midpoint (H) V1.13" /home/rnebot/GoogleDrive/AA_SENTINEL/dev_lcia_methods/LCIA_implementation_3.7.1.xlsx "/home/rnebot/GoogleDrive/AA_SENTINEL/dev_nis_case_studies/library (Base)/reduced_lcia_methods_recipe_h_v113.csv"

        :param lcia_implementation_file: The full path of the INPUT LCIA implementation XLSX file
        :param lcia_file: The full path of the OUTPUT CSV file
        :param method_like: To filter rows, let pass only those where method contains "method_like"
        :param method_is: To filter rows, let pass only those where method is in "method_is"
        :param include_obsolete: To not filter rows, where method is considered obsolete
        :param use_nis_name_syntax: Transform names which are used as NIS identifiers to the NIS identifier syntax
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :return:
        """
        set_log_level_from_cli_param(log)
        if isinstance(method_is, str):
            method_is = [method_is]
        convert_lcia_implementation_to_nis(lcia_implementation_file, lcia_file,
                                           method_like, method_is,
                                           include_obsolete, use_nis_name_syntax)

    # def lci_to_interface_types(self, lci_file_path: str, csv_file: str):
    #     """
    #     Extract the interface types from an LCI file
    #
    #     :param lci_file_path: Path of a single .spold file
    #     :param csv_file: Output CSV with a listing of distinct interface types (potentially importable in InterfaceTypes)
    #     :return:
    #     """
    #     lci_to_interfaces_csv(lci_file_path, csv_file)

    @staticmethod
    def lci_to_nis(nis_base_url: str,
                   spold_files_folder: str,
                   nis_structurals_output: str, log: str = None):
        """
        Scan "NIS base file" for references to Spold files (that should be available in "Spold files folder"), and
        elaborate another NIS file at "NIS structurals output", with equivalent MuSIASEM structural Processors.
        "enbios enviro" needs the information to be incorporated into the NIS base file, so the NIS Base file should be
        modified to import the elaborated NIS file, using an ImportCommands command.

        :param spold_files_folder: Local folder where .spold files are stored
        :param nis_base_url: URL of a NIS file that will be used as Base for the assembly of the model
        :param nis_structurals_output: File name of the output NIS that will contain the structural processors
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :return:
        """
        # The resulting NIS file has these commands:
        #    - InterfaceTypes
        #    - BareProcessors
        #    - Interfaces

        set_log_level_from_cli_param(log)
        s2n = SpoldToNIS()


        # TODO CORRESPONDENCE_PATH not USED (NIS base file contains all the information)
        #  The correspondence file should be a CSV file with the following header:
        #     name,match_target_type,match_target,weight,match_conditions
        #  Example:
        #     enbios lci_to_nis /home/rnebot/GoogleDrive/AA_SENTINEL/enviro_tmp/ /home/rnebot/GoogleDrive/AA_SENTINEL/enviro/sim_correspondence_example.csv https://docs.google.com/spreadsheets/d/15NNoP8VjC2jlhktT0A8Y0ljqOoTzgar8l42E5-IRD90/edit?usp=sharing /home/rnebot/Downloads/out.xlsx
        #     enbios lci_to_nis /home/rnebot/GoogleDrive/AA_SENTINEL/enviro_tmp/ "" https://docs.google.com/spreadsheets/d/1nYzphq1XYrezquW3yj7UiJ8sNJ8RzJ8gqErg3oVmw2Q/edit?usp=sharing /home/rnebot/Downloads/lci_to_nis_output.xlsx
        #
        #  Example lines in this file:
        #     wind_onshore_competing,lca,2fdb6da1-8281-453d-8a7c-0184dc3586c4_66c93e71-f32b-4591-901c-55395db5c132.spold,,
        #     wind_onshore_competing,musiasem,Energy_system.Electricity_supply.Electricity_generation.Electricity_renewables.Electricity_wind.Electricity_wind_onshore,,

        correspondence_path = ""
        s2n.spold2nis("generic_energy_production",
                      spold_files_folder,
                      correspondence_path,
                      nis_base_url,
                      nis_structurals_output)

    @staticmethod
    def sentinel_to_nis_preparatory(sentinel_data_package_json_path: str, nis_file: str, log: str = None):
        """
        Read a Sentinel Data Package and elaborate an XLSX file with NIS commands and complementary information
        describing what is inside the Sentinel Package (enumeration of Scenarios, Regions, etc.) and a template of
        CORRESPONDENCE file as one of the Worksheets.

        :param sentinel_data_package_json_path: The full path of the JSON file describing a Sentinel data package
        :param nis_file: Full path of the output XLSX NIS formatted file
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :return:
        """
        set_log_level_from_cli_param(log)
        sentinel_to_prep_file(sentinel_data_package_json_path, nis_file)

    @staticmethod
    def list_lcia_indicators(nis_file: str, output_file: str, log: str = None):
        """
        Parse the NIS file and extract a list of LCIA indicators

        :param nis_file: Full path of the NIS file (local or Google Drive, .XLSX format)
        :param output_file: Full path of the output .XLSX file
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :return:
        """
        set_log_level_from_cli_param(log)
        list_lcia_indicators(nis_file, output_file)

    @staticmethod
    def enviro(cfg_file_path,
               just_prepare_base: bool = False,
               fragments_list_file: bool = False,
               first_fragment: int = 0,
               n_fragments: int = 0,
               max_lci_interfaces: int = 0,
               keep_min_fragment_files: bool = True,
               generate_nis_base_file: bool = False,
               generate_full_fragment_files: bool = False,
               generate_interface_results: bool = False,
               generate_indicators: bool = False,
               n_cpus: int = 1,
               log: str = None,
               ):
        """
        The main function of the package, reads the contents of the configuration file, which can be either a JSON or YAML
        file. Following an example of YAML file contents:
            nis_file_location: "https://docs.google.com/spreadsheets/d/12AlJ0tdu2b-cfalNzLqFYfiC-hdDlIv1M1pTE3AfSWY/edit#gid=1986791705"
            correspondence_files_path: ""
            simulation_type: sentinel
            simulation_files_path: ""
            output_directory: ""

        (you can copy and paste the example in an empty text file as reference, and ".yaml" as extension)

        :param cfg_file_path: Path of the configuration file in YAML format
        :param n_fragments: number of fragments to process, 0 for "all"
        :param first_fragment: Index of the first fragment to be processed. To obtain an ordered list of fragments, execute first "enbios enviro" with "--just-prepare-base --fragments-list-file" options
        :param generate_nis_base_file: True if the Base file should be generated (once) for testing purposes
        :param generate_full_fragment_files: True to generate a full NIS formatted XLSX file for each fragment
        :param generate_interface_results: True to generate a CSV with values at interfaces for each fragment
        :param generate_indicators: True to generate a CSV with indicators for each fragment
        :param fragments_list_file: True to generate a CSV with the list of fragments
        :param keep_min_fragment_files: If True, do not delete minimal NIS files submitted to NIS to compute indicators
        :param max_lci_interfaces: Max number of LCI interfaces to consider. 0 for all (default 0)
        :param n_cpus: Number of CPUs of the local computer used to perform the process (default 1, sequential; 0 to find a good value for the computer automatically)
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :param just_prepare_base: True to only prepare (download, parse and execute, then cache; but not solve) Base file and exit
        :return:
        """
        # locale.setlocale(locale.LC_ALL, 'en_US')
        set_log_level_from_cli_param(log)
        t = Enviro()
        t.set_cfg_file_path(cfg_file_path)

        t.compute_indicators_from_base_and_simulation(n_fragments,
                                                      first_fragment,
                                                      generate_nis_base_file,
                                                      generate_full_fragment_files,
                                                      generate_interface_results,
                                                      keep_min_fragment_files,
                                                      generate_indicators,
                                                      fragments_list_file,
                                                      max_lci_interfaces,
                                                      n_cpus,
                                                      just_prepare_base)

    @staticmethod
    def visualize(cfg_file_path, open_browser=False, log: str = None):
        """
        Prepare a Dash server enabling basic visualization of results
            - scenario
            - region
            - year
            - processor
            - [carrier]
            - indicator

        :param cfg_file_path:
        :param open_browser: If True, open the link in the browser
        :param log:
        :return:
        """
        # Read data
        cfg_file_path = os.path.realpath(cfg_file_path)
        cfg = read_parse_configuration(cfg_file_path)
        base_dir = cfg["output_directory"]
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(os.path.dirname(cfg_file_path), base_dir)

        app = create_dashboard_app(base_dir)

        # Launch Dash dashboard server
        host = "localhost"
        port = 8050
        address = f"http://{host}:{port}"
        print(f"Please, open {address}. Ctrl+C to Stop.")
        if open_browser:
            webbrowser.open(address)
        app.run_server(host=host, port=port, use_reloader=False)


def main():
    import platform
    os.environ["PAGER"] = "cat" if platform.system().lower() != "windows" else "-"
    fire.Fire(Enbios)


if __name__ == '__main__':
    main()
