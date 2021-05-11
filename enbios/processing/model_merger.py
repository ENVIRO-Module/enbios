import pandas as pd
from nexinfosys.embedded_nis import NIS
from typing import Dict
from enbios.input import Simulation
from enbios.input.lci import LCIIndex
from enbios.processing.nis_file_generator import generate_import_commands_command, generate_processors_command, \
    generate_interfaces_command


def block_meets_conditions(block, condition):
    """
    Parse expression with
    :param block:
    :param condition:
    :return:
    """
    # TODO arith_boolean_expression_with_less_tokens (DESPUÉS DE NEXINFOSYS V0.47)
    from nexinfosys.command_generators.parser_field_parsers import string_to_ast, arith_boolean_expression
    from nexinfosys.model_services import State
    from nexinfosys.command_generators.parser_ast_evaluators import ast_evaluator
    try:
        ast = string_to_ast(arith_boolean_expression, condition)
    except:
        print(f"A syntax error here")
    s = State()
    for k, v in block.items():
        s.set(k, v)
    issues = []
    res, unres = ast_evaluator(ast, s, None, issues)
    if isinstance(res, bool):
        return res
    else:
        return unres  # It is an error probably


class Matcher:
    """
    * Match Activities (LCA) and Technologies (simulation)
    * Match Technologies (simulation) into parent Processors (MuSIASEM)
    """
    def __init__(self, block_types_path, correspondence_path):
        self._block_types_path = block_types_path
        self._correspondence_path = correspondence_path
        self._typed_block = {}
        self._sim_lca = {}  # Store a list of target LCA activities for each simulation type
        self._sim_musiasem = {}  # Store a list of target MuSIASEM processors for each simulation type
        self._read()

    def _read(self):
        """ Read the correspondence file(s). Format:
technology,match_target_type,match_target,weight,match_conditions
wind_onshore_competing,lca,2fdb6da1-8281-453d-8a7c-0184dc3586c4_66c93e71-f32b-4591-901c-55395db5c132.spold,,
wind_onshore_competing,musiasem,Energy_system.Electricity_supply.Electricity_generation.Electricity_renewables.Electricity_wind.Electricity_wind_onshore,,
...
        """
        # ### Read block types (if defined) ###
        self._typed_block = {}
        df = pd.read_csv(self._block_types_path)
        # Assume all columns are present: "name", "type"
        for idx, r in df.iterrows():
            if self._typed_block.get(r["name"]):
                print(f"Entry for {r['name']} repeated. Overwriting")
            self._typed_block[r["name"]] = r["type"]

        # ### Read correspondence ###
        self._sim_lca = {}
        self._sim_musiasem = {}
        df = pd.read_csv(self._correspondence_path)
        # Assume all columns are present: "name", "match_target_type", "match_target", "weight", "match_conditions"
        for idx, r in df.iterrows():
            if r["match_target_type"].lower() == "lca":
                d = self._sim_lca
            else:
                d = self._sim_musiasem
            lst = d.get(r["name"], [])
            if len(lst) == 0:
                d[r["name"]] = lst
            lst.append(dict(target=r["match_target"], weight=r["weight"], conditions=r["match_conditions"]))

    def find_related_lci_activities(self, block: Dict[str, str], lci_index: LCIIndex):
        block_name = block["name"]
        lst = self._sim_lca.get(block_name)
        if not lst:
            block_type = self._typed_block.get(block)
            if block_type:
                return self.find_related_lci_activities(block, lci_index)
        else:
            lst2 = []
            for b in lst:
                if block_meets_conditions(block, b["match_conditions"]):
                    lst2.append(b)
            return [dict(activity=lci_index.read_activity(b["target"]), weight=b["weight"]) for b in lst2]

    def find_candidate_musiasem_parent_processors(self, nis: NIS, block):
        # TODO Find parent processors names in the correspondence, locate them in NIS and return the full hierarchy name
        #      of each
        pass

    def equivalent_interface_type_for_lci_flow(self, lca_flow: str, nis: NIS):
        pass

    def equivalent_interface_type_for_simulation_flow(self, simulation_flow: str, nis: NIS):
        pass


def merge_models(nis: NIS, matcher: Matcher, simulation: Simulation, lci_data_index: LCIIndex):
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
        matching_lci_activities = matcher.find_related_lci_activities(block, lci_data_index)
        for lci_activity in matching_lci_activities:
            for activity_flow in lci_activity.flows:
                # TODO Name, value, unit, input/output, time, location, source, etc.
                block_info.update()
                # TODO Take note of "Impact Category" (analysis) -> Attribute "@ImpactCategory"

        # TODO Read "LCIA method" -> Schema of SUM (exchanges*weight) per Impact Category

        # TODO Convert input information into NIS InterfaceTypes (including change of scale?)
        block_info = matcher.convert_to_interface_types(block_info)
        # Matching MuSIASEM processors
        parents = matcher.find_candidate_musiasem_parent_processors(nis, block)
        # TODO obtain full processor names
        block_info["parents"] = parents
        # Append one more processor
        blocks_as_processor_templates.append(block_info)
    # Generate NIS commands
    lst = []
    lst.append(("Import", generate_import_commands_command([])))  # Input NIS file path
    # Enumerate the blocks and relate them to parent processors
    lst.append(("BareProcessors", generate_processors_command(blocks_as_processor_templates)))
    lst.append(("Interfaces", generate_interfaces_command(blocks_as_processor_templates)))
    # TODO Dump
    #  blocks with no MuSIASEM parents (as warning?),
    #  blocks with no matching LCI (as warning)
    #  flows with no matching InterfaceType (as error)
    return lst