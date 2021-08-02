# /usr/bin/env python
import locale
import os
import fire
import webbrowser

from nexinfosys.bin.cli_script import set_log_level_from_cli_param
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
    def lcia_implementation_to_csv(lcia_implementation_file: str, lcia_file: str, log: str = None):
        """
        Convert an XLSX file with LCIA implementation (from Ecoinvent) to a CSV file ready to be declared in a
        DatasetDef which can later be imported in a "LCIAMethod" command

        Example:
           enbios lcia_implementation_to_csv /home/rnebot/GoogleDrive/AA_SENTINEL/LCIA_implementation_3.7.1.xlsx /home/rnebot/GoogleDrive/AA_SENTINEL/case_studies/library/lcia_implementation_nis.csv


        :param lcia_implementation_file: The full path of the input LCIA implementation XLSX file
        :param lcia_file: The full path of the output CSV file
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :return:
        """
        set_log_level_from_cli_param(log)
        convert_lcia_implementation_to_nis(lcia_implementation_file, lcia_file)

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
    def lci_to_nis(spold_files_folder: str, correspondence_path: str,
                   nis_base_url: str, nis_structurals_base_path: str, log: str = None):
        """
        Scan correspondence file AND a NIS file to extract a NIS file with the commands:
           - InterfaceTypes
           - BareProcessors
           - Interfaces
        The correspondence file should be a CSV file with the following header:
           name,match_target_type,match_target,weight,match_conditions
        Example:
           enbios lci_to_nis /home/rnebot/GoogleDrive/AA_SENTINEL/enviro_tmp/ /home/rnebot/GoogleDrive/AA_SENTINEL/enviro/sim_correspondence_example.csv https://docs.google.com/spreadsheets/d/15NNoP8VjC2jlhktT0A8Y0ljqOoTzgar8l42E5-IRD90/edit?usp=sharing /home/rnebot/Downloads/out.xlsx
           enbios lci_to_nis /home/rnebot/GoogleDrive/AA_SENTINEL/enviro_tmp/ "" https://docs.google.com/spreadsheets/d/1nYzphq1XYrezquW3yj7UiJ8sNJ8RzJ8gqErg3oVmw2Q/edit?usp=sharing /home/rnebot/Downloads/lci_to_nis_output.xlsx

        Example lines in this file:
           wind_onshore_competing,lca,2fdb6da1-8281-453d-8a7c-0184dc3586c4_66c93e71-f32b-4591-901c-55395db5c132.spold,,
           wind_onshore_competing,musiasem,Energy_system.Electricity_supply.Electricity_generation.Electricity_renewables.Electricity_wind.Electricity_wind_onshore,,


        :param spold_files_folder: Local folder where .spold files are stored
        :param correspondence_path: Correspondence file
        :param nis_base_url: URL of a NIS file that will be used as Base for the assembly of the model
        :param nis_structurals_base_path: File name of the output NIS that will contain the structural processors
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :return:
        """
        set_log_level_from_cli_param(log)
        s2n = SpoldToNIS()
        s2n.spold2nis("generic_energy_production", spold_files_folder, correspondence_path, nis_base_url, nis_structurals_base_path)

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
    def enviro(cfg_file_path,
               just_one_fragment: bool = False,
               generate_nis_base_file: bool = False,
               generate_full_fragment_files: bool = False,
               generate_interface_results: bool = False,
               generate_indicators: bool = False,
               keep_min_fragment_files: bool = True,
               max_lci_interfaces: int = 0,
               n_cpus: int = 1,
               log: str = None,
               just_prepare_base: bool = False):
        """
        The main function of the package, reads the contents of the configuration file, which can be either a JSON or YAML
        file. Following an example of YAML file contents:
            nis_file_location: "https://docs.google.com/spreadsheets/d/12AlJ0tdu2b-cfalNzLqFYfiC-hdDlIv1M1pTE3AfSWY/edit#gid=1986791705"
            correspondence_files_path: ""
            simulation_type: sentinel
            simulation_files_path: ""
            output_directory: ""

        (you can copy and paste the example in an empty text file as reference, and ".yaml" as extension)

        :param cfg_file_path:
        :param just_one_fragment: True if only one of the fragments is to be computed, to test things
        :param generate_nis_base_file: True if the Base file should be generated (once) for testing purposes
        :param generate_full_fragment_files: True if the current fragment should be dumped into a NIS formatted XLSX file
        :param generate_interface_results: True if a CSV with values at interfaces should be produced, for each fragment
        :param generate_indicators: True if a CSV with indicators should be produced, for each fragment
        :param keep_min_fragment_files: If True, do not delete minimal NIS files submitted to NIS to compute indicators
        :param max_lci_interfaces: Max number of LCI interfaces to consider. 0 for all (default 0)
        :param n_cpus: Number of CPUs of the local computer used to perform the process (default 1, sequential)
        :param log: Set log level to one of: Error (E, Err), Debug (D), Warning (W, Warn), Info (I), Off, Critical (Fatal)
        :param just_prepare_base: True to only preparing Base file and exit
        :return:
        """
        # locale.setlocale(locale.LC_ALL, 'en_US')
        set_log_level_from_cli_param(log)
        t = Enviro()
        t.set_cfg_file_path(cfg_file_path)

        t.compute_indicators_from_base_and_simulation(just_one_fragment,
                                                      generate_nis_base_file,
                                                      generate_full_fragment_files,
                                                      generate_interface_results,
                                                      keep_min_fragment_files,
                                                      generate_indicators,
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
