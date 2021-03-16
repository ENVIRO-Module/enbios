import calliope

from typing import Dict, List, Tuple

from nexinfosys.command_generators import Issue
from nexinfosys.embedded_nis import NIS



class LCIIndex:
    """ An index of where is each activity, to ease reading it """
    def __init__(self, lci_data_locations: List[str]):
        # TODO Prepare an index of activities, and where each is stored. Maybe also the flows
        pass



class Matcher:
    """ * Match Technologies (simulation) into parent Processors (MuSIASEM)
        * Match Activities (LCA) into Technologies (simulation)
    """
    def __init__(self, correspondence_files_path):
        self.correspondence_location = correspondence_files_path
        self._read()

    def _read(self):
        """ Read the correspondence files """
        # TODO
        pass

    def find_lci_activities(self, block, lci_index: LCIIndex):
        # TODO
        pass

    def find_parent_processors(self, nis: NIS, block):
        # TODO
        pass


class Simulation:
    """ Abstract simulation reader (the implementation will depend on the type of simulator used) """
    def read(self, simulation_files_path: str):
        """ Read"""
        pass

    def blocks(self):
        """ An iterator into the blocks (technologies) in the simulation """
        pass

    def block_information(self, block_name):
        """ Return information for block """
        pass


class CalliopeSimulation(Simulation):
    def __init__(self, simulation_files_path):
        self._model = None
        self.read(simulation_files_path)

    def read(self, simulation_files_path: str):
        self._model = calliope.Model(simulation_files_path)

    def blocks(self):
        """ List of technologies """

    def block_information(self, block_name):
        """ Return information for block """
        pass


def build_model(nis: NIS, matcher: Matcher, simulation: Simulation, lci_data_index: LCIIndex):
    """
    Inputs:
      - Assessment configuration
      - Base MuSIASEM NIS file
      - LCI data
      - Simulation
        - Structure
        - Inputs
        - Outputs
      - Correspondence file
    :return:
    """
    # TODO - Parse all inputs
    #        - Base MuSIASEM: read, have model
    #        - Simulation structure
    #        - Correspondence file:
    #          - NIS: every "leaf" processor is in the file
    #          - Simulation: every technology is in the file, and can be assigned to at least one processor (more than one?)
    #          - LCA: every technology has an LCI
    #      - Configuration: * SPACE. single system or system per location or per country?
    #                       * TIME. Single NIS or multiple iterations
    #      - Merge NIS file with Simulation and LCI data
    #      - Compute indicators. Data structure: a dataset (dimensions - values - attributes) for each multilevel, another for scalar
    #      - Generate: jupyter script Python-R with example graphs. Network graph?, Sankey?, Geolayer?
    blocks_as_processor_templates = []
    for block in simulation.blocks:
        # Information (interfaces) of block from the simulation
        # TODO Name, value, unit, input/output, time, location, source, etc.
        block_info = simulation.read_block_info(block)
        # Matching LCA information
        matching_lci_activities = matcher.find_lci_activities(block, lci_data_index)
        for lci_activity in matching_lci_activities:
            for activity_flow in lci_activity.flows:
                # TODO Name, value, unit, input/output, time, location, source, etc.
                block_info.update()
        # TODO Convert input information into NIS InterfaceTypes (including change of scale?)
        block_info = matcher.convert_to_interface_types(block_info)
        # Matching MuSIASEM processors
        parents = matcher.find_parent_processors(nis, block)
        # TODO obtain full processor names
        # block_info["parents"] = processor_names(parents)
        # Append one more processor
        blocks_as_processor_templates.append(block_info)
    # TODO Generate NIS commands
    # processors_command = generate_processors_command(blocks_as_processor_templates)
    # interfaces_command = generate_interfaces_command(blocks_as_processor_templates)
    # TODO Dump
    #  blocks with no MuSIASEM parents (as warning?),
    #  blocks with no matching LCI (as warning)
    #  flows with no matching InterfaceType (as error)
    pass


def read_parse_configuration(file_name) -> Dict:
    def read_yaml_configuration(file_content) -> Dict:
        import yaml
        d = yaml.load(file_content, Loader=yaml.FullLoader)
        return d

    # Read into string
    with open(file_name, 'r') as file:
        contents = file.read().replace('\n', '')

    # Read depending on format
    if file_name.endswith("yaml"):
        cfg = read_yaml_configuration(contents)

    # Check configuration
    mandatory_keys = ["nis_file_location",
                      "correspondence_files_path",
                      "simulation_type", "simulation_files_path",
                      "lci_data_locations"]
    any_not_found = False
    for k in mandatory_keys:
        if k not in cfg:
            print(f"{k} key not found in the configuration file {file_name}")
            any_not_found = True
    if any_not_found:
        raise Exception("Configuration file incomplete or incorrect. See log for missing keys")
    return cfg


def read_prepare_nis_file(nis_file_url: str) -> Tuple[NIS, List[Issue]]:
    nis = NIS()
    nis.open_session(True)
    nis.reset_commands()
    # Load file and execute it
    nis.load_workbook(nis_file_url)
    issues = nis.submit_and_solve()
    # tmp = nis.query_available_datasets()
    # print(tmp)
    # d = nis.get_results([(tmp[0]["type"], tmp[0]["name"])])
    return nis, issues


def assemble(cfg_file_path):
    """
    From the configuration file, read all inputs
      - Base NIS format file
      - Correspondence file
      - Simulation type, simulation location
      - LCI databases

    :param configuration_file:
    :return:
    """
    cfg = read_parse_configuration(cfg_file_path)
    # NIS
    nis = read_prepare_nis_file(cfg["nis_file_location"])
    # Matcher
    matcher = Matcher(cfg["correspondence_files_path"])
    # Simulation
    if cfg["simulation_type"].lower() == "calliope":
        simulation = CalliopeSimulation(cfg["simulation_files_path"])
    # LCI index
    lci_data_index = LCIIndex(cfg["lci_data_locations"])

    build_model(nis, matcher, simulation, lci_data_index)
