# /usr/bin/env python
import os
import fire

from enbios.input.data_preparation.lci_to_nis import lci_to_interfaces_csv, SpoldToNIS
from enbios.input.data_preparation.recipe_to_nis import convert_recipe_to_nis
from enbios.input.data_preparation.sentinel_to_nis_prep import sentinel_to_prep_file
from enbios.processing.main import Enviro

"""
DEVELOPMENT CLI EXECUTION:
 
python -m enbios.bin.enbios
  <it will show the automatically generated help>

CLI EXECUTION:
script.py (same as previous)
  <it will show the automatically generated help>
  
"""


class EnbiosCLI:
    def recipe_to_nis(self, recipe_file: str, lcia_file: str):
        """
        Convert an XLSX file with ReCiPe2016 to a CSV file ready to be declared in a DatasetDef which can be imported
        in a "LCIAMethod" command

        :param recipe_file: The full path of the input ReCiPe2016 XLSX file
        :param lcia_file: The full path of the output CSV file
        :return:
        """
        convert_recipe_to_nis(recipe_file, lcia_file)

    def sentinel_to_nis_preparatory(self, sentinel_data_package_json_path: str, nis_file: str):
        """
        Read a Sentinel Data Package and elaborate an XLSX file with NIS commands and complementary information
        describing what is inside the Sentinel Package (enumeration of Scenarios, Regions, etc.) and a template of
        CORRESPONDENCE file as one of the Worksheets.

        :param sentinel_data_package_json_path: The full path of the JSON file describing a Sentinel data package
        :param nis_file: Full path of the output XLSX NIS formatted file
        :return:
        """
        sentinel_to_prep_file(sentinel_data_package_json_path, nis_file)

    def lci_to_interface_types(self, lci_file_path: str, csv_file: str):
        """
        Extract the interface types from an LCI file

        :param lci_file_path: Path of a single .spold file
        :param csv_file: Output CSV with a listing of distinct interface types (potentially importable in InterfaceTypes)
        :return:
        """
        lci_to_interfaces_csv(lci_file_path, csv_file)


    # lci_to_nis /home/rnebot/GoogleDrive/AA_SENTINEL/enviro_tmp/ /home/rnebot/GoogleDrive/AA_SENTINEL/enviro/sim_correspondence_example.csv https://docs.google.com/spreadsheets/d/15NNoP8VjC2jlhktT0A8Y0ljqOoTzgar8l42E5-IRD90/edit?usp=sharing /home/rnebot/Downloads/out.xlsx
    def lci_to_nis(self, spold_files_folder: str, correspondence_path: str, nis_base_url: str, nis_structurals_base_path: str):
        """
        Scan correspondence file AND a NIS file to extract a NIS file with the commands:
           - InterfaceTypes
           - BareProcessors
           - Interfaces
        The correspondence file should be a CSV file with the following header:
           name,match_target_type,match_target,weight,match_conditions

        Example lines in this file:
           wind_onshore_competing,lca,2fdb6da1-8281-453d-8a7c-0184dc3586c4_66c93e71-f32b-4591-901c-55395db5c132.spold,,
           wind_onshore_competing,musiasem,Energy_system.Electricity_supply.Electricity_generation.Electricity_renewables.Electricity_wind.Electricity_wind_onshore,,

        :param spold_files_folder: Local folder where .spold files are stored
        :param correspondence_path: Correspondence file
        :param nis_base_url: URL of a NIS file that will be used as Base for the assembly of the model
        :param nis_structurals_base_path: File name of the output NIS that will contain the structural processors
        :return:
        """
        s2n = SpoldToNIS()
        s2n.spold2nis(spold_files_folder, correspondence_path, nis_base_url, nis_structurals_base_path)

    def enviro(self, cfg_file_path):
        """
        The main function of the package, reads the contents of the configuration file, which can be either a JSON or YAML
        file. Following an example of YAML file contents:
            nis_file_location: "https://docs.google.com/spreadsheets/d/12AlJ0tdu2b-cfalNzLqFYfiC-hdDlIv1M1pTE3AfSWY/edit#gid=1986791705"
            correspondence_files_path: ""
            simulation_type: sentinel
            simulation_files_path: ""
            lci_data_locations:
              - ""
            output_directory: ""

        (you can copy and paste the example in an empty text file as reference, and ".yaml" as extension)

        :param cfg_file_path:
        :return:
        """
        t = Enviro()
        t.set_cfg_file_path(cfg_file_path)
        t.compute_indicators_from_base_and_simulation()


def main():
    os.environ["PAGER"] = "cat"
    fire.Fire(EnbiosCLI)


if __name__ == '__main__':
    main()