import tempfile

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
    def __init__(self, cfg_file_path):
        self.set_cfg_file_path(cfg_file_path)

    def set_cfg_file_path(self, cfg_file_path):
        self._cfg_file_path = cfg_file_path

    def musiasem(self):
        """
        From the configuration file, read all inputs
          - Base NIS format file
          - Correspondence file
          - Simulation type, simulation location
          - LCI databases

        :param configuration_file:
        :return:
        """
        # Read configuration
        cfg = read_parse_configuration(self._cfg_file_path)

        # Construct auxiliary models
        # Matcher
        matcher = Matcher(cfg["correspondence_files_path"])
        # MuSIASEM (NIS)
        nis = read_prepare_nis_file(cfg["nis_file_location"])
        # LCI index
        lci_data_index = LCIIndex(cfg["lci_data_locations"])
        # Simulation
        if cfg["simulation_type"].lower() == "calliope":
            simulation = CalliopeSimulation(cfg["simulation_files_path"])
        elif cfg["simulation_type"].lower() == "sentinel":
            simulation = SentinelSimulation(cfg["simulation_files_path"])

        # ELABORATE MERGE NIS FILE
        lst = merge_models(nis, matcher, simulation, lci_data_index)

        # Generate NIS file into a temporary file
        temp_name = tempfile.NamedTemporaryFile(dir=cfg["output_directory"], delete=False)
        generate_workbook_from_list_of_worksheets(temp_name, lst)

        # Execute the NIS file
        nis, issues = read_prepare_nis_file(temp_name)

        # TODO Download outputs
        # TODO Elaborate indicators
        # TODO Write indicator files

        # os.remove(temp_name)


# model = CalliopeSimulation("/home/rnebot/GoogleDrive/AA_SENTINEL/calliope_tests/Calliope-Kenya/model.yaml")


