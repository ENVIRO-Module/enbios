import tempfile

import pandas as pd
from typing import Tuple

from nexinfosys.embedded_nis import NIS

from enbios.input import Simulation
from enbios.input.data.lci import LCIIndex
from enbios.input.simulators.calliope import CalliopeSimulation
from enbios.input.simulators.sentinel import SentinelSimulation
from enbios.processing import read_parse_configuration, read_prepare_nis_file
from enbios.processing.model_merger import Matcher, merge_models
from enbios.processing.nis_file_generator import generate_workbook_from_list_of_worksheets


#####################################################
# MAIN ENTRY POINT  #################################
#####################################################
class Enviro:
    def __init__(self):
        self._cfg_file_path = None
        self._cfg = None

    def set_cfg_file_path(self, cfg_file_path):
        self._cfg = read_parse_configuration(cfg_file_path)
        self._cfg_file_path = cfg_file_path if isinstance(cfg_file_path, str) else None

    def _prepare_process(self) -> Tuple[NIS, LCIIndex, Simulation]:
        # Construct auxiliary models
        # Simulation
        if self._cfg["simulation_type"].lower() == "calliope":
            simulation = CalliopeSimulation(self._cfg["simulation_files_path"])
        elif self._cfg["simulation_type"].lower() == "sentinel":
            simulation = SentinelSimulation(self._cfg["simulation_files_path"])
        # MuSIASEM (NIS)
        nis = read_prepare_nis_file(self._cfg["nis_file_location"])
        # LCI index
        lci_data_index = LCIIndex(self._cfg["lci_data_locations"])

        return nis, lci_data_index, simulation

    def generate_matcher_templates(self, combine_countries=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate auxiliary DataFrames to help in the elaboration of:
          - correspondence file
          - block types file
        :param combine_countries:
        :return: A tuple with 4 pd.DataFrames
        """
        # Construct auxiliary models
        nis, lci_data_index, simulation = self._prepare_process()

        # TODO Build a DataFrame with a list of LCI title, to file code
        #   name,title,file
        lci_df = pd.DataFrame()
        # TODO Build a DataFrame with a list of BareProcessors
        #   processor,parent
        nis_df = pd.DataFrame()
        # TODO Build DataFrame "correspondence"
        #   name,match_target_type,match_target,weight,match_conditions
        #   OK   "lca"             ---          1      <generate>
        #  if combine_countries:
        #   for each <name> generate a line per country and a line without country
        correspondence_df = pd.DataFrame()
        # TODO Build DataFrame "block_types"
        #   name,type
        #   OK   <from an existing database of tech names to Calliope tech types>
        block_types_df = pd.DataFrame()
        return lci_df, nis_df, correspondence_df, block_types_df

    def musiasem_indicators(self, system_per_country=True):
        """
        From the configuration file, read all inputs
          - Base NIS format file
          - Correspondence file
          - Simulation type, simulation location
          - LCI databases

        :param system_per_country:
        :return:
        """
        # Construct auxiliary models
        nis, lci_data_index, simulation = self._prepare_process()
        # Matcher
        matcher = Matcher(self._cfg["correspondence_files_path"])

        # ELABORATE MERGE NIS FILE
        lst = merge_models(nis, matcher, simulation, lci_data_index)

        # Generate NIS file into a temporary file
        temp_name = tempfile.NamedTemporaryFile(dir=self._cfg["output_directory"], delete=False)
        generate_workbook_from_list_of_worksheets(temp_name, lst)

        # Execute the NIS file
        nis, issues = read_prepare_nis_file(temp_name)

        # TODO Download outputs
        # TODO Elaborate indicators
        # TODO Write indicator files

        # os.remove(temp_name)


t = Enviro()
_ = dict(nis_file_location="https://docs.google.com/spreadsheets/d/12AlJ0tdu2b-cfalNzLqFYfiC-hdDlIv1M1pTE3AfSWY/edit#gid=1986791705",
         correspondence_files_path="",
         simulation_type="sentinel",
         simulation_files_path="/home/rnebot/Downloads/borrame/calliope-output/datapackage.json",
         lci_data_locations={},
         output_directory="/home/rnebot/Downloads/borrame/enviro-output/")
t.set_cfg_file_path(_)
d1, d2, d3, d4 = t.generate_matcher_templates()
print("Hola")

# model = CalliopeSimulation("/home/rnebot/GoogleDrive/AA_SENTINEL/calliope_tests/Calliope-Kenya/model.yaml")


