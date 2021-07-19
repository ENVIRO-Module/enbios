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
