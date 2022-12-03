import functools
import logging
import operator
import os
import sys
import time
from multiprocessing import Pool, cpu_count
import itertools

import pandas as pd
from typing import Tuple, Dict, Set, List, Optional
from NamedAtomicLock import NamedAtomicLock
from friendly_data.converters import from_df
from nexinfosys.bin.cli_script import get_valid_name, get_file_url, prepare_base_state, print_issues
from nexinfosys.command_generators import IType, Issue
from nexinfosys.command_generators.parser_ast_evaluators import ast_evaluator
from nexinfosys.command_generators.parser_field_parsers import string_to_ast, arith_boolean_expression

from nexinfosys.common.helper import PartialRetrievalDictionary, any_error_issue, create_dictionary
from nexinfosys.embedded_nis import NIS
from nexinfosys.model_services import State
from nexinfosys.models.musiasem_concepts import Processor, ProcessorsRelationPartOfObservation, Indicator, \
    FactorQuantitativeObservation
from nexinfosys.serialization import deserialize_state

from enbios.common.helper import generate_workbook, list_to_dataframe, get_scenario_name
from enbios.input import Simulation
from enbios.input.lci import LCIIndex
from enbios.input.simulators.sentinel import SentinelSimulation
from enbios.input.simulators.single_csv import AllInACSV
from enbios.model import SimStructuralProcessorAttributes, g_default_subtech
from enbios.processing import read_parse_configuration, read_submit_solve_nis_file


#####################################################
# MAIN ENTRY POINT  #################################
#####################################################
_idx_cols = ["region", "scenario", "subscenario", "technology", "subtechnology", "model", "carrier",
             "year", "time", "timestep", "unit", "variable", "description"]


def parallelizable_process_fragment(param: Tuple[str,  # Fragment label
                                                 Dict[str, str],  # Fragment dict
                                                 Dict[str, Set[str]],  #
                                                 List[SimStructuralProcessorAttributes]],
                                    s_state: bytes,
                                    output_dir: str,
                                    development_nis_file: str,
                                    generate_nis_fragment_file: bool,
                                    generate_interface_results: bool,
                                    generate_indicators: bool,
                                    max_lci_interfaces: int,
                                    keep_fragment_file: bool
                                    ):
    """
    Prepares a NIS file from inputs and submits it to NIS for accounting and calculation of indicators

    :param param: A Tuple with the information to drive the process
    :param s_state: A "bytes" with Serialized state (deserialized inside)
    :param output_dir: Outputs directory
    :param development_nis_file: URL of a NIS file that would go after state and before the fragment,
                                parsed everytime (for experiments)
    :param generate_nis_fragment_file: True to generate an expanded NIS file for the fragment
    :param generate_interface_results: True to generate a NIS file with the values of interfaces after solving
    :param generate_indicators: True to generate a file with all indicators
    :param max_lci_interfaces: If >0, cut the LCI interfaces used to that number
    :param keep_fragment_file: If True, keep the minimal fragment NIS file
    :return:
    """

    def write_outputs(outputs):
        """
        Write results of a submission to output directory

        :param outputs: Dictionary with the possible outputs
        :return:
        """
        nis_idempotent_file = outputs.get("idempotent_nis", None)
        df_indicators = outputs.get("indicators", pd.DataFrame())
        df_global_indicators = outputs.get("global_indicators", pd.DataFrame())
        df_interfaces = outputs.get("interfaces", pd.DataFrame())
        df_iamc_indicators = outputs.get("iamc_indicators", pd.DataFrame())
        print("Writing results ...")

        def append_fragment_to_file(file_name, df):
            lock = NamedAtomicLock("enbios-lock")
            lock.acquire()
            try:
                full_file_name = os.path.join(output_dir, file_name)
                if not os.path.isfile(full_file_name):
                    df.to_csv(full_file_name, index=False)
                else:
                    df.to_csv(full_file_name, index=False, mode='a', header=False)
            finally:
                lock.release()

        # Main result: "indicators.csv" file (aggregates all fragments)
        if df_indicators is not None:
            append_fragment_to_file("indicators.csv", df_indicators)

            # Write a separate indicators.csv for the fragment
            if generate_indicators:
                csv_name = os.path.join(output_dir, f"fragment_indicators{get_valid_name(str(p_key))}.csv")
                if not df_indicators.empty:
                    df_indicators.to_csv(csv_name, index=False)
                else:
                    with open(csv_name, "wt") as f:
                        f.write("Could not obtain indicator values (??)")

        # "system_global_indicators.csv" file (aggregates all fragments)
        if df_global_indicators is not None:
            append_fragment_to_file("system_global_indicators.csv", df_global_indicators)

        if df_iamc_indicators is not None:
            append_fragment_to_file("iamc_indicators.csv", df_iamc_indicators)

        # Write NIS of the fragment just processed
        if generate_nis_fragment_file and nis_idempotent_file is not None:
            fragment_file_name = os.path.join(output_dir, f"full_fragment{get_valid_name(str(p_key))}.xlsx")
            with open(fragment_file_name, "wb") as f:
                f.write(nis_idempotent_file)

        # Write Dataset with values for each interface as calculated by NIS solver, for the fragment
        if generate_interface_results and df_interfaces is not None:
            csv_name = os.path.join(output_dir, f"fragment_interfaces{get_valid_name(str(p_key))}.csv")
            if not df_interfaces.empty:
                df_interfaces.to_csv(csv_name, index=False)
            else:
                with open(csv_name, "wt") as f:
                    f.write("Could not obtain interface values (??)")

    # Starts here
    frag_label, p_key, f_metadata, f_processors = param  # Unpack "param"
    print(f"Fragment processing ...")
    start = time.time()  # Time execution

    # Call main function
    outputs = process_fragment(s_state, p_key,
                               f_metadata, f_processors,
                               output_dir,
                               development_nis_file,
                               max_lci_interfaces,
                               keep_fragment_file,
                               generate_nis_fragment_file,
                               generate_interface_results)
    end = time.time()  # Stop timing
    print(f"Fragment processed in {end - start} seconds ---------------------------------------")
    write_outputs(outputs)
    start = time.time()  # Time also output writing
    print(f"Fragment outputs written in {start-end} seconds to output dir: {output_dir}")


class Enviro:
    def __init__(self):
        self._cfg_file_path = None
        self._cfg = None
        self._simulation_files_path = None

    def set_cfg_file_path(self, cfg_file_path):
        self._cfg = read_parse_configuration(cfg_file_path)
        self._cfg_file_path = os.path.realpath(cfg_file_path) if isinstance(cfg_file_path, str) else None
        if "simulation_files_path" in self._cfg:
            self._simulation_files_path = self._cfg["simulation_files_path"]

    def _get_simulation(self) -> Simulation:
        # Simulation
        simulation = None
        if self._cfg["simulation_type"].lower() == "sentinel":
            simulation = SentinelSimulation(self._simulation_files_path)
        elif self._cfg["simulation_type"].lower() == "single_csv":
            simulation = AllInACSV(self._simulation_files_path)

        # elif self._cfg["simulation_type"].lower() == "calliope":
        #     simulation = CalliopeSimulation(self._simulation_files_path)
        return simulation

    def _prepare_process(self) -> Tuple[NIS, LCIIndex, Simulation]:
        # Simulation
        simulation = self._get_simulation()
        # MuSIASEM (NIS)
        nis, issues = read_submit_solve_nis_file(self._cfg["nis_file_location"], state=None, solve=False)
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

    def _read_simulation_fragments(self, split_by_region=True, split_by_scenario=True, split_by_period=True, default_time=None):

        """
        Read simulation. Find fragments and then start an iteration on them
        Each fragment is made of the "processors" and their "interfaces"
        It is here where basic processors are built. Correspondence file has tech but it could also
        have carrier and region

        :return:
        """
        # Create simulation
        simulation = self._get_simulation()
        # Read Simulation, returning:
        # prd: PartialRetrievalDictionary
        # scenarios: list of scenarios
        # regions: list of regions
        # times: list of time periods
        # carriers: list of carriers
        # units: list of units
        # col_types: list of value (not index) fields
        # ctc: list of pairs (tech, region, carrier)
        prd, scenarios, regions, times, techs, carriers, units, col_types, ctc = simulation.read("", default_time)

        partition_lists = []
        mandatory_attributes = ["carrier", "technology"]
        if split_by_region and len(regions) > 0:
            partition_lists.append([("_g", r) for r in regions])
            mandatory_attributes.append("region")
        if split_by_period and len(times) > 0:
            partition_lists.append([("_d", t) for t in times])
            mandatory_attributes.append("time")
        # TODO Spores
        if split_by_scenario and len(scenarios) > 0:
            partition_lists.append([("_s", s) for s in scenarios])
            mandatory_attributes.append("scenario")

        for i, partition in enumerate(sorted(list(itertools.product(*partition_lists)))):
            partial_key = {t[0]: t[1] for t in partition}
            procs = prd.get(partial_key)
            # Sweep "procs" and update periods, regions, scenarios, models and carriers
            md = dict(periods=set(), regions=set(), scenarios=set(), models=set(), carriers=set(), techs=set(), flows=set())
            accountable_procs = []
            for p in procs:
                any_mandatory_not_found = False
                for mandatory in mandatory_attributes:
                    if mandatory not in p.attrs:
                        # print(f"Processor {p.attrs} did not define '{mandatory}'. Skipping.")
                        any_mandatory_not_found = True
                        break
                if any_mandatory_not_found:
                    continue
                accountable_procs.append(p)
                if "region" in p.attrs:
                    md["regions"].add(p.attrs["region"])
                if "scenario" in p.attrs:
                    md["scenarios"].add(p.attrs["scenario"])
                if "technology" in p.attrs:
                    md["techs"].add(p.attrs["technology"])
                if "time" in p.attrs:
                    md["periods"].add(p.attrs["time"])
                if "model" in p.attrs:
                    md["models"].add(p.attrs["model"])
                if "carrier" in p.attrs:
                    md["carriers"].add(p.attrs["carrier"])
                tmp = set(p.attrs.keys()).difference(_idx_cols)
                md["flows"].update(tmp)
            # Return 4 items:
            #  - a fragment label, to sort fragments
            #  - an equivalent (to the label) dict
            #  - fragment metadata
            #  - the Processors (the data to process!)
            # print(':'.join([f"{k}{v}" for k, v in partial_key.items()]))

            yield ':'.join([f"{k}{v}" for k, v in partial_key.items()]), partial_key, md, accountable_procs

    def compute_indicators_from_base_and_simulation(self,
                                                    n_fragments: int = 0,
                                                    first_fragment: int = 0,
                                                    generate_nis_base_file: bool = False,
                                                    generate_nis_fragment_file: bool = False,
                                                    generate_interface_results: bool = False,
                                                    keep_fragment_file: bool = True,
                                                    generate_indicators: bool = False,
                                                    fragments_list_file: bool = False,
                                                    max_lci_interfaces: int = 0,
                                                    n_cpus: int = 1,
                                                    just_prepare_base: bool = False):
        """
        MAIN entry point of current ENVIRO
        Previously, a Base NIS must have been prepared, see @_prepare_base

        :param n_fragments: number of fragments to process, 0 for "all"
        :param first_fragment: Index of the first fragment to be processed. To obtain an ordered list of fragments, execute first "enbios enviro" with --fragments-list-file option
        :param generate_nis_base_file: True if the Base file should be generated (once) for testing purposes
        :param generate_nis_fragment_file: True to generate a full NIS formatted XLSX file for each fragment
        :param generate_interface_results: True to generate a CSV with values at interfaces for each fragment
        :param generate_indicators: True to generate a CSV with indicators for each fragment
        :param fragments_list_file: True to generate a CSV with the list of fragments
        :param max_lci_interfaces: Max number of LCI interfaces to consider. 0 for all (default 0)
        :param n_cpus: Number of CPUs of the local computer used to perform the process
        :param just_prepare_base: True to only prepare (download, parse and execute, then cache; but not solve) Base file and exit
        :param keep_fragment_file: If True, do not remove the minimal NIS file generated and submitted to NIS
        :return:
        """

        output_dir = self._cfg["output_directory"]
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self._cfg_file_path), output_dir)

        print(f"output_dir: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare Base
        state, serial_state, issues = prepare_base_state(self._cfg["nis_file_location"], False, output_dir)

        if not state:
            logging.debug(f"NIS BASE PREPARATION FAILED")
            print_issues("NIS base preparation", self._cfg["nis_file_location"], issues,
                         f"Base not ready due to errors, exiting. Please check the issues")
            sys.exit(1)

        if just_prepare_base and not fragments_list_file:
            print("Base processed and cached, exiting because 'just_prepare_base == True'")
            sys.exit(1)

        if "development_nis_file_location" in self._cfg:
            development_nis_file = self._cfg["development_nis_file_location"]
        else:
            development_nis_file = None

        # [Write Base as a full NIS file]
        if generate_nis_base_file:
            issues, new_state, results = run_nis_for_results(None, None, state, ["model"])
            nis_file = results[0]
            with open(os.path.join(output_dir, "nis_base.idempotent.xlsx"), "wb") as f:
                f.write(nis_file)

        # Read simulation AND Split it in decoupled fragments
        default_time = self._cfg.get("simulation_default_time")
        fragments = sorted([_ for _ in self._read_simulation_fragments(default_time=default_time)],
                           key=operator.itemgetter(0))
        fragments = [_ for _ in fragments if len(_[3]) > 0]

        # Produce file with an enumeration of fragments
        if fragments_list_file:
            possible_columns = dict(_g="Regions", _d="Periods", _s="Scenarios")
            used_columns = []
            used_columns_set = set()
            for fragment in fragments:
                for key in fragment[1].keys():
                    if key not in used_columns_set:
                        used_columns.append(key)
                        used_columns_set.add(key)
            s = [", ".join([possible_columns[_] for _ in used_columns])]
            for fragment in fragments:
                s.append(", ".join([f"{fragment[1].get(_, '')}" for _ in used_columns]))
            with open(os.path.join(output_dir, "fragments_list.csv"), "w") as f:
                f.write("\n".join(s))

        logging.debug(f"Simulation read, and split in {len(fragments)} fragments")

        if just_prepare_base:
            print("Base processed and cached, exiting because 'just_prepare_base == True'")
            sys.exit(1)

        # Remove "indicators.csv" and "iamc_indicators if they exist
        for f_name in ["indicators.csv", "iamc_indicators.csv"]:
            f_path = os.path.join(output_dir, f_name)
            if os.path.isfile(f_path):
                os.remove(f_path)

        # Possibly reduce the list of fragments to process
        if n_fragments > 0 or first_fragment > 0:
            if n_fragments == 0:
                fragments = fragments[first_fragment:]
            else:
                fragments = fragments[first_fragment:first_fragment + n_fragments]
            logging.debug(f"{n_fragments} fragment{'s' if n_fragments > 1 else ''} will be processed, starting from fragment {first_fragment}")

        # PROCESS FRAGMENTS - Sequential or embarrassingly parallel
        # If 0 -> find an appropriate number of CPUs to use. int(80% of # CPUs) if more than 4; 2 for 4; 1 for < 4
        if n_cpus == 0:
            n_cpus = int(0.8*cpu_count()) if cpu_count() > 4 else 2 if cpu_count() == 4 else 1

        if n_cpus == 1:
            for i, (frag_label, partial_key, frag_metadata, frag_processors) in enumerate(fragments):
                # fragment_metadata: dict with regions, years, scenarios in the fragment
                # fragment_processors: list of processors with their attributes which will be interfaces
                # print(f"{partial_key}: {len(frag_processors)}")
                parallelizable_process_fragment((frag_label, partial_key, frag_metadata, frag_processors), serial_state,
                                                output_dir,
                                                development_nis_file,
                                                generate_nis_fragment_file, generate_interface_results,
                                                generate_indicators, max_lci_interfaces, keep_fragment_file)
        else:
            p = Pool(n_cpus)
            p.map(functools.partial(parallelizable_process_fragment,
                                    s_state=serial_state,
                                    output_dir=output_dir,
                                    development_nis_file=development_nis_file,
                                    generate_nis_fragment_file=generate_nis_fragment_file,
                                    generate_interface_results=generate_interface_results,
                                    generate_indicators=generate_indicators,
                                    max_lci_interfaces=max_lci_interfaces, keep_fragment_file=keep_fragment_file),
                  fragments)


def run_nis_for_results(nis_file_name: Optional[str],
                        development_nis_file: Optional[str],
                        _state: State,
                        requested_outputs: List[Tuple] = None):
    """
    Call to NIS and return results

    :param nis_file_name: File to process
    :param development_nis_file: URL of a NIS file that would go after state and before the fragment,
                                parsed everytime (for experiments)
    :param _state: Initial State assumed by the file to process
    :param requested_outputs:
    :return: Issues, new State, and either: a) the requested outputs (None entry if error) or
                                            b) a list of available outputs
    """
    outputs = {
        "flow_graph_solution": ("dataset", "flow_graph_solution", "csv"),
        "flow_graph_solution_indicators": ("dataset", "flow_graph_solution_indicators", "csv"),
        "flow_graph_solution_edges": ("dataset", "flow_graph_solution_edges", "csv"),
        "flow_graph_global_indicators": ("dataset", "flow_graph_global_indicators", "csv"),
        "flow_graph_solution_benchmarks": ("dataset", "flow_graph_solution_benchmarks", "csv"),
        "flow_graph_solution_global_benchmarks": ("dataset", "flow_graph_solution_global_benchmarks", "csv"),
        "benchmarks_and_stakeholders": ("dataset", "benchmarks_and_stakeholders", "csv"),
        "model": ("model", "Model.xlsx"),
        "interfaces_graph": ("graph", "interfaces_graph.visjs"),
        "processors_graph": ("graph", "processors_graph.visjs"),
        "processors_geolayer": ("geolayer", "processors_geolayer.geojson"),
    }
    nis = NIS()
    nis.open_session(True, _state)
    if nis_file_name:
        if development_nis_file:
            nis.load_workbook(development_nis_file)
        nis.load_workbook(get_file_url(nis_file_name))
        issues = nis.submit_and_solve()
        error = any_error_issue(issues)
        new_state = nis.get_state()
    elif isinstance(_state, State):
        # nis.append_command("Ignore me", list_to_dataframe([["dummy"], ["dummy"]]))
        issues = []
        error = False
        new_state = _state
    else:
        issues = [Issue(itype=IType.ERROR,
                        description=f"At least NIS file or initial State must be specified",
                        location=None)]
        error = True
        new_state = None
    if not error:
        if requested_outputs:
            _ = []
            for r in requested_outputs:
                if isinstance(r, str):
                    if r in outputs:
                        _.append(outputs[r])
                    else:
                        issues.append(Issue(itype=IType.ERROR,
                                            description=f"Dataset '{r}' not registered",
                                            location=None))
                elif isinstance(r, tuple):
                    _.append(r)
            _ = [r[0] if r[2] else None for r in nis.get_results(_)]
        else:
            _ = nis.query_available_datasets()
    else:
        _ = []

    nis.close_session()

    return issues, new_state, _


def process_fragment(base_serial_state: bytes,
                     partial_key: Dict[str, str],
                     fragment_metadata: Dict[str, Set[str]],
                     fragment_processors: List[SimStructuralProcessorAttributes],
                     output_directory: str,
                     development_nis_file: str,
                     max_lci_interfaces: int,
                     keep_fragment_file: bool,
                     generate_nis_fragment_file: bool,
                     generate_interface_results: bool
                     ):
    """
    :param base_serial_state: A "bytes" with Serialized state (deserialized inside)
    :param partial_key: A dictionary describing the fragment dimensions
    :param fragment_metadata: A dictionary with sets of dimensions inside the fragment
    :param fragment_processors: A list of the structural processor attributes (read from simulation output)
                                inside the fragment.
    :param output_directory: Outputs directory
    :param development_nis_file: URL of a NIS file that would go after state and before the fragment,
                                parsed everytime (for experiments)
    :param max_lci_interfaces: If >0, cut the LCI interfaces used to that number
    :param keep_fragment_file: If True, do not remove the minimal fragment file submitted to NIS
    :param generate_nis_fragment_file: If True generate the idempotent NIS model
    :param generate_interface_results: If True generate the interface results (used to do the indicator calculations)

    Different assembly options
     * Dendrogram functional processors: a clone per region / a fragment per region (so no need to clone)
    Spold:
     * Structural processors: currently there is a list of structural (not-accounted) processors attached to the
     last level of functional processors, with a .spold file associated with them.
     When a simulation processor is found, associate it with one of these processors (equal "tech" name)
     * If there is a .spold file associated to the structural processor associated to the simulation, use it
     * If there is no .spold file linked to this processor and the "parent processor" has one:
       - attach the simulation processor to the parent processor
       - use the spold file of the parent

     Something like:
          if spold(Sim) is None
            if spold(P(Sim)) -> spold(Sim) = spold(P(Sim))
            else ERR
          else OK
          INSERT Sim interfaces
          output_interface_name <- name of the output interface from interfaces in the simulation processor
          for spold_proc in spold(Sim)
            for each interface in spold_proc
              INSERT interface, make them relative_to "output_interface_name"
          COPY all interfaces of spold(Sim) to Sim (insert "interfaces") and make them relative_to the output interface

          if skip_aux_structural AND is_structural(P(Sim)) AND is_not_accounted(P(Sim)) -> P(Sim) = P(P(Sim))
    """

    # Declare (InterfaceType) the original (from simulation) name even if it is renamed (EcoinventCarrierName) and
    # duplicate the Interface with the original and the changed name
    write_original_interface = True

    def _get_processors_by_type():
        """
        Returns two sets of Processors:
          - "Functional without Interfaces": MuSIASEM dendrogram (without data, so no top-down would be possible)
          - "Structural not Accounted": LCI processors and Simulation processors

        NOTE: these two sets are not a partition of all Processors: we may have "Functional that may have Interfaces"
              and "Structural which are Accounted"

        The implicit input is the state in "prd" (PartialRetrievalDictionary)
        :return: The two sets
        """
        procs = prd.get(Processor.partial_key())
        musiasem_dendrogram_procs = {}
        structural_procs = {}
        for proc in procs:
            n_interfaces = len(proc.factors)
            accounted = proc.instance_or_archetype.lower() == "instance"
            functional = proc.functional_or_structural.lower() == "functional"
            if accounted and n_interfaces == 0:
                for h_name in proc.full_hierarchy_names(prd):
                    musiasem_dendrogram_procs[h_name] = proc
            elif not functional and not accounted:
                for h_name in proc.full_hierarchy_names(prd):
                    structural_procs[h_name] = proc
            else:
                print(f"{proc.full_hierarchy_names(prd)[0]} is neither MuSIASEM base nor LCI base "
                      f"(functional: {functional}; accounted: {accounted}; n_interfaces: {n_interfaces}.")
        return musiasem_dendrogram_procs, structural_procs

    def _find_parents(registry: PartialRetrievalDictionary, base_procs: Dict[str, Processor], p: Processor):
        """
        Find names of parent Processors of "p" (not all ancestors, just parents).

        :param registry: Registry of NIS objects
        :param base_procs: Dict of Processor in the dendrogram
        :param p: name of a simulation process, which should be enumerated in the previous Dict
        :return: Dict of parents, {processor_name: Processor}. None if "p" is not found
        """
        return [rel.parent_processor
                for rel in registry.get(ProcessorsRelationPartOfObservation.partial_key(child=p))]

        # Find the Processor object
        # SimProc(tech, carrier) ->(child-of)-> FunctMuSProc(ParentProc)
        # SimProc(tech, carrier) as StrucMuSProc(tech, carrier2)

    def _find_lci_and_tech_processors(reg: PartialRetrievalDictionary, structural_procs, base_procs, p) -> Dict[Processor, List]:
        """
        Find LCI Processor(s) which are assimilated to "p"
          - Processor "p" should exist, as Structural/not-Accounted, with a Parent, [with no Interfaces?]
          - Ecospold file potentially associated
          - if "p" does not have an Ecospold file associated, take parent's Ecospold file
          - Find an "LCI processor": Structural/not-Accounted, NO Parent, with the same Ecospold

        :param reg: Registry of NIS objects
        :param structural_procs: Dict of Structural not Accounted Processors, which includes LCI processors but
                                 also Simulation processors
        :param base_procs: Dict of Dendrogram Processors
        :param p: Name of the target processor
        :return: List of matching LCI processors, None if "p" is not found; tech processor tech output variable name, desired output name (carrier name)
        """
        def _find_reference_lci_processor(target: Processor, lci_procs: Dict[str, Processor]):
            lci_proc = None
            ecospold_file = target.attributes.get("EcoinventFilename", "")
            if ecospold_file != "":
                for proc_name, proc in lci_procs.items():
                    if "." not in proc_name:
                        ecospold_file_2 = proc.attributes.get("EcoinventFilename", "")
                        if len(ecospold_file) > 10 and len(ecospold_file_2) > 10 and \
                           (ecospold_file.lower() in ecospold_file_2.lower() or
                           ecospold_file_2.lower() in ecospold_file.lower()):
                            lci_proc = proc
                            break
            return lci_proc

        # Find structural not accounted matching the sim processor (match tech name and proc carrier name)
        targets: Dict[Processor, List] = {}
        tech = p.attrs["technology"].lower()
        carrier_ = p.attrs["carrier"].lower()  # Mandatory
        for proc_name, proc in structural_procs.items():
            last = proc_name.split(".")[-1]
            if tech == last.lower():
                if carrier_ != "":
                    carrier_field = ""
                    if "SimulationCarrier" in proc.attributes:
                        carrier_field = "SimulationCarrier"
                    elif "EcoinventCarrierName" in proc.attributes:
                        carrier_field = "EcoinventCarrierName"  # It should never enter here
                    if proc.attributes.get(carrier_field, "").lower() == carrier_:
                        targets[proc] = []
                else:
                    targets[proc] = []  # If "tech" matches and "carrier_" is ""

        for target in targets.keys():
            lci_proc = _find_reference_lci_processor(target, structural_procs)
            if lci_proc is None:
                # Parent
                s1 = target.full_hierarchy_names(reg)[0].rsplit(".", 1)[0]
                for proc_name, proc in base_procs.items():
                    if s1 == proc_name:
                        lci_proc = _find_reference_lci_processor(proc, structural_procs)
                        targets[target].append(lci_proc)
                        break
            else:
                targets[target].append(lci_proc)

        return targets

    def _generate_fragment_nis_file(clone_processors, cloners, processors, interfaces, variables, scenarios):
        """
        Once model has been assembled in memory, generate a NIS file
        This file can be later submmited for it to be accounted and indicators calculated

        :param clone_processors:
        :param cloners:
        :param processors:
        :param interfaces:
        :param variables:
        :param scenarios:
        :return:
        """
        cmds = []
        # ProblemStatement
        lst = [["Scenario", "Parameter", "Value", "Description"]]
        for s in scenarios:
            # Scenario, Parameter, Value, Description
            scen = get_scenario_name("s", s)
            observer = get_scenario_name("o", s)
            lst.append([scen, "NISSolverObserversPriority", observer, f"Scenario {scen}, observer {observer}"])
        cmds.append(("ProblemStatement sim fragment", list_to_dataframe(lst)))
        # RefProvenance
        lst = [["RefID", "ProvenanceFileURL", "AgentType", "Agent", "Activities", "Entities"]]
        for s in scenarios:
            observer = get_scenario_name("o", s)
            scen = get_scenario_name("s", s)
            lst.append([observer, "", "Software", f"Sentinel simulation observer for scenario {scen}", "WP4", ""])
        cmds.append(("RefProvenance sim fragment", list_to_dataframe(lst)))
        # InterfaceTypes
        lst = [["InterfaceTypeHierarchy", "InterfaceType", "Sphere", "RoegenType", "ParentInterfaceType", "Formula",
                "Description", "Unit", "OppositeSubsystemType", "Attributes"]]
        for v_name, attrs in variables.items():
            if not attrs["main_flow"]:
                lst.append(["Sentinel", v_name, "Technosphere", "Flow", "", "", "", attrs["unit"], "", ""])
        cmds.append(("InterfaceTypes sim fragment", list_to_dataframe(lst)))

        if clone:
            cmds.append(("BareProcessors regions", list_to_dataframe(clone_processors)))
            cmds.append(("ProcessorScalings", list_to_dataframe(cloners)))
        cmds.append(("BareProcessors sim fragment", list_to_dataframe(processors)))
        cmds.append(("Interfaces sim fragment", list_to_dataframe(interfaces)))
        s = generate_workbook(cmds)
        if s:
            frag_file_name = os.path.join(output_directory, f"fragment_{get_valid_name(str(partial_key))}.xlsx")
            with open(frag_file_name, "wb") as f:
                f.write(s)
            return frag_file_name
        else:
            return None

    def is_main_flow(flow_name, output_flow="flow_out_sum"):  # TODO Depends on simulation
        return flow_name in [output_flow]

    def get_flow_orientation(flow_name):  # TODO Depends on simulation
        return "Output" if "out" in flow_name else "Input"

    def get_flow_unit(flow_name):  # TODO Depends on simulation
        if flow_name.lower() in ["flow_out", "flow_in", "flow_out_sum", "flow_in_sum"]:
            return "kWh"
        elif flow_name.lower() in ["capacity_factor"]:
            return "kW"
        else:
            return "dimensionless"

    def _interface_used_in_some_indicator(iface):
        # TODO Implement
        return True

    def _add_clone(cloners_list, clone_processors_list, regions, base_musiasem_procs):
        cloners_list.append(
            ["InvokingProcessor", "RequestedProcessor", "ScalingType", "InvokingInterface", "RequestedInterface",
             "Scale"])
        clone_processors_list.append(
            ["ProcessorGroup", "Processor", "ParentProcessor", "SubsystemType", "System", "FunctionalOrStructural",
             "Accounted", "Stock", "Description", "GeolocationRef", "GeolocationCode", "GeolocationLatLong",
             "Attributes",
             "@SimulationName", "@Region"])
        # For each region, create a processor in "clone_processors" and an entry in "cloners" hanging the top
        # Processors in the dendrogram into the region processor
        for reg in regions:
            clone_processors_list.append(["", reg, "", "", reg, "Functional", "Yes", "",
                                         f"Country level processor, region '{reg}'", "", "", "", "", reg, reg])
            considered = set()
            for p_name, proc in base_musiasem_procs.items():
                if len(p_name.split(".")) > 1 and p_sim not in considered:
                    considered.add(proc)
                    # InvokingInterface, RequestedInterface, Scale are mandatory; specify something here
                    cloners_list.append((reg, p_name, "Clone", "", "", ""))

    def _get_processor_name(p):
        return f"{p.attrs['technology']}_{p.attrs['carrier']}"

    def _split_processor_name(p, carriers, techs):
        """
        Split a processor name into technology and carrier, in accordance with "_get_processor_name"

        :param p:
        :param carriers:
        :param techs:
        :return:
        """
        # Metadata contains technologies and carriers, which are code lists
        r_ = p.split(".")
        if r_[-1].endswith(tuple(carriers)) and r_[-1].startswith(tuple(techs)):
            # Find carrier and split
            # (carriers must be sorted by length, to avoid cases like this. If the carrier is "syn_diesel" and
            #  we have ["diesel", "syn_diesel"], it will fail. But if "carriers" is sorted by length, it will
            #  split "r_" properly)
            for c in carriers:
                if r_[-1].endswith(c):
                    nc = len(c)
                    return f"{'.'.join(r_[:-1])}.{r_[-1][:-nc-1]}", r_[-1][-nc:]
        else:
            return p, ""

    def _add_processor(proc, matches, processors_list, already_added_processors_set, techs_used_in_regions, context):
        """
        Inserts elements in "processors_list", directly usable as worksheet rows in a BareProcessors command

        :param proc:
        :param matches:
        :param processors_list:
        :param already_added_processors_set:
        :param techs_used_in_regions:
        :param context:
        :return:
        """
        first_name = None
        return_simple = True
        # Add "Accounted Structural" Processors, hanging from each parent
        for parent_p in matches:
            # NEW PROCESSOR NAME: technology + carrier [+ region]
            p_name = _get_processor_name(proc)
            if p_name in techs_used_in_regions:
                return_simple = False
            else:
                techs_used_in_regions.add(p_name)
            parent_name = parent_p.name
            description = f"Simulation {proc.attrs['technology']} for {proc.attrs['carrier']} at region {region}, scenario {scenario}."
            original_name = f"{proc.attrs['technology']}"
            if first_name is None:
                first_name = f"{parent_name}.{p_name}"
                if (p_name, parent_name) not in already_added_processors_set:
                    processors_list.append(["", p_name, parent_name, "", region, "Structural", "Yes", "",
                                            description, "", "", "", "", original_name, region])
                    already_added_processors_set.add((p_name, parent_name))
            else:
                if (p_name, parent_name) not in already_added_processors_set:
                    processors_list.append(["", first_name, parent_name, "", region, "Structural", "Yes", "",
                                            description, "", "", "", "", original_name, region])
                    already_added_processors_set.add((p_name, parent_name))
        return first_name, parent_name  # p_name if return_simple else first_name, parent_name

    def _add_interfaces(proc, proc_name, proc_parent_name, lci_matches, matching_tech_proc, context) -> None:
        """
        Inserts elements in "interfaces", directly usable as worksheet rows in an Interfaces command

        :param proc:
        :param proc_name:
        :param proc_parent_name:
        :param lci_matches: Which LCI processors match
        :param matching_tech_proc: Which technology processor matches
        :param context: "time", "region", "scenario", ... in which interfaces are being added
        :return:
        """

        def get_tech_simulation_scaling_factor():
            _ = matching_tech_proc.attributes.get("SimulationScalingFactor", "1.0")
            ast = string_to_ast(arith_boolean_expression, _)
            state = State(context)
            issues = []
            res, variables = ast_evaluator(ast, state, None, issues)
            if res:
                return res
            else:
                raise ValueError(f"Could not evaluate {_}. Variables: {variables}. Issues: {issues}")

        relative_to = None
        tech_output_name = matching_tech_proc.attributes.get("SimulationVariable")
        tech_desired_output_name = matching_tech_proc.attributes.get("EcoinventCarrierName")
        tech_simulation_scaling_factor = get_tech_simulation_scaling_factor()
        tech_output_to_spold_factor = float(matching_tech_proc.attributes.get("SimulationToEcoinventFactor", "1.0"))
        observer = get_scenario_name("o", scenario)
        # Interfaces from MuSIASEM tech
        for i in matching_tech_proc.factors:
            for o in i.observations:
                if isinstance(o, FactorQuantitativeObservation):
                    interfaces.append([name, i.name, "", "", "", i.orientation, "", "", "", "", o.value,
                                       "", "", "", "", "", "", o.attributes["time"],
                                       o.observer.name if o.observer else "", "", ""])

        # Interfaces from simulation
        for i in set(proc.attrs.keys()).difference(_idx_cols):
            i_name = i
            input_i_name = i_name
            v = proc.attrs[i]
            if is_main_flow(i, tech_output_name):
                if tech_desired_output_name:
                    i_name = tech_desired_output_name
                v *= tech_output_to_spold_factor * tech_simulation_scaling_factor  # Change scale
                relative_to = i_name  # All LCI interfaces will be relative to main output
                orientation = "Output"
            else:
                orientation = get_flow_orientation(i)
            interfaces.append([proc_name, i_name, "", "", "", orientation, "", "", "", "", v,
                               "", "", "", "", "", "", time_, observer, "", ""])
            if write_original_interface and input_i_name != i_name:
                interfaces.append([proc_name, input_i_name, "", "", "", orientation, "", "", "", "", v,
                                   "", "", "", "", "", "", time_, observer, "", ""])
            if i_name not in variables:
                variables[i_name] = dict(main_flow=is_main_flow(i), orientation=orientation, unit=get_flow_unit(i))
            if write_original_interface and input_i_name not in variables:
                variables[input_i_name] = dict(main_flow=False, orientation=orientation, unit=get_flow_unit(i))

        if lci_matches[0] is None or len(lci_matches) == 0:
            print(f"{proc.attrs} does not have an LCI processor associated. "
                  f"Cannot provide information for indicators, skipped.")
            return

        # Interfaces from LCI
        # The following condition could happen if the fragment includes several Regions, Years and Scenarios.
        # LCI does not change -currently, regionalized LCI would be different-
        if (proc_name, proc_parent_name) not in already_added_lci:
            already_added_lci.add((proc_name, proc_parent_name))

            for cp in lci_matches:
                cont = 0
                for i in cp.factors:
                    if cont >= max_lci_interfaces:
                        break  # Artificial limit to reduce file size, for test purposes
                    if not _interface_used_in_some_indicator(
                            i.name):  # Economy of the model: avoid specifying interfaces not used later
                        continue
                    q = i.quantitative_observations[0]
                    if q.attributes.get("relative_to", None) is None:
                        continue
                    cont += 1
                    unit = q.attributes.get("unit")  # TODO unit / relative_to_unit
                    orientation = i.attributes.get("orientation", "")
                    v = q.value
                    # TODO Static: does not depend on scenario or time, avoid inserting repeatedly
                    interfaces.append([proc_name, i.taxon.name, i.name, "", "", orientation, "", "", "", "", v,
                                       "", relative_to, "", "", "", "", "Year", "Ecoinvent", "", ""])

    def _prepare_iamc_dataframe_fragment(state: State, indicators: pd.DataFrame, iamc_codes: Dict[str, str], metadata):
        # self._simulation_files_path
        # TODO Write the indicators using Sentinel-friendly-data-IAMC format
        #  from friendly_data.io import dwim_file (Do What I Mean)
        #  from friendly_data.iamc import IAMconv
        #  from friendly_data.dpkg import pkgindex
        #  conf = dwim_file("conf.yaml")
        #  conf["indices"]
        #  idx = pkgindex.from_file("index.yaml")
        #  converter = IAMconv(idx, conf["indices"], basepath="<location>")
        #  conf2 = conf["indices"]
        #  conf2["year"] = 2050
        #  converter2 = IAMconv(idx, conf2, basepath="")
        #  converter.to_csv o to_df
        #  .
        #  Also, e-mail from Suvayu
        # try:
        #     res = from_df(df_indicators, output_dir, os.path.join(output_dir, f"indicators_fd.csv"))
        # except:
        #     traceback.print_exc()
        iamc_df = indicators.copy(True)
        # Scope
        iamc_df = iamc_df[iamc_df["Scope"] == "Internal"]
        iamc_df = iamc_df.drop(columns=["Scope"])
        # Model
        iamc_df["Model"] = "enbios[calliope]"
        # Scenario, Region, Value, Unit, Year
        iamc_df.rename(columns=dict(System="Region", Scenario="Scenario", Value="Value", Unit="Unit", Period="Year"),
                       inplace=True)

        def _get_iamc_code(row, prd, iamc_codes, carriers, techs):
            processor, indicator = row["Processor"], row["Indicator"]
            # First part of the code, from the processor
            if processor in iamc_codes:
                first_part = iamc_codes[processor]
            else:
                parts = _split_processor_name(processor, carriers, techs)
                p = prd.get(Processor.partial_key(parts[0]))
                if p and len(p) == 1:
                    first_part = p[0].attributes.get("iamccode", "")
                else:
                    first_part = ""
            # Second part of the code, from the indicator
            ind = prd.get(Indicator.partial_key(indicator))
            if ind and len(ind) == 1:
                # TODO It could also be that if "IAMCCodeSuffix" was not specified, the second part is left empty,
                #   so the row is skipped
                second_part = ind[0].attributes.get("iamccodesuffix", indicator)
            else:
                second_part = indicator
            # Combine both parts and return
            if first_part and second_part:
                return f"{first_part}|{second_part}"
            else:
                return ""

        def _get_indicator_unit(row, prd):
            indicator = row["Indicator"]
            ind = prd.get(Indicator.partial_key(indicator))
            if ind and len(ind) == 1:
                return ind[0]._unit_label if ind[0]._unit_label else ind[0]._unit
            else:
                return ""

        specially_sorted_carriers = sorted(metadata["carriers"], key=len, reverse=True)
        iamc_df["Variable"] = iamc_df.apply(
            functools.partial(_get_iamc_code,
                              prd=state.get("_glb_idx"),
                              iamc_codes=iamc_codes_techs,
                              carriers=specially_sorted_carriers,
                              techs=metadata["techs"]),
            axis=1)
        iamc_df["Unit"] = iamc_df.apply(
            functools.partial(_get_indicator_unit,
                              prd=state.get("_glb_idx")),
            axis=1)

        iamc_df = iamc_df[iamc_df["Variable"] != ""]

        # Rearrange columns
        iamc_df = iamc_df[["Model", "Scenario", "Region", "Variable", "Unit", "Value", "Year", "Processor"]]

        return iamc_df

    # MAIN - Integration, NIS file for the fragment, and NIS submission (Indicators) -----------------------------------
    print(f"Processing fragment {fragment_metadata}")

    # PREPARE / INITIALIZE ------------------
    state = deserialize_state(base_serial_state)
    prd = state.get("_glb_idx")
    dendrogram_musiasem_procs, structural_not_accounted_procs = _get_processors_by_type()

    processors = [
        ["ProcessorGroup", "Processor", "ParentProcessor", "SubsystemType", "System", "FunctionalOrStructural",
         "Accounted", "Stock", "Description", "GeolocationRef", "GeolocationCode", "GeolocationLatLong", "Attributes",
         "@SimulationName", "@Region"]]  # TODO Maybe more fields to qualify the fragment
    interfaces = [
        ["Processor", "InterfaceType", "Interface", "Sphere", "RoegenType", "Orientation", "OppositeSubsystemType",
         "GeolocationRef", "GeolocationCode", "InterfaceAttributes", "Value", "Unit", "RelativeTo", "Uncertainty",
         "Assessment", "PedigreeMatrix", "Pedigree", "Time", "Source", "NumberAttributes", "Comments"]]
    techs_used_in_regions = set()  # Register if a technology has appeared in a region. Used to generate complete names
    already_added_processors = set()
    already_added_lci = set()
    variables = dict()
    max_lci_interfaces = 100000 if max_lci_interfaces == 0 else max_lci_interfaces
    # If no periods are defined, define a specific year. At least one is needed for the solver to run
    default_time = "2038" if len(fragment_metadata["periods"]) == 0 else "Year"

    # CLONE ------------------
    # Two approaches "clone" or "not clone". "clone" would be to have multi-region models. For now, "not clone"
    clone = False
    cloners = []
    clone_processors = []
    if clone:
        _add_clone(cloners, clone_processors, fragment_metadata["regions"], dendrogram_musiasem_procs)
    else:
        # Modify all parent processors system to region "r"
        if len(fragment_metadata["regions"]) == 1:
            r = next(iter(fragment_metadata["regions"]))
            for p_sim in dendrogram_musiasem_procs.values():
                p_sim.processor_system = r

    # FOR EACH SIMULATION PROCESSOR (of the fragment)
    iamc_codes_techs = create_dictionary()
    for p_sim in fragment_processors:
        # Update time and scenario
        # (they can change from entry to entry, but for MuSIASEM the same functional Processor is used)
        time_ = p_sim.attrs.get("time", p_sim.attrs.get("year", default_time))
        scenario = p_sim.attrs.get("scenario", "")
        region = p_sim.attrs.get("region", "")
        carrier = p_sim.attrs.get("carrier", "")
        tech = p_sim.attrs.get("technology", "")
        subtech = p_sim.attrs.get("subtechnology", g_default_subtech)
        subscenario = p_sim.attrs.get("subscenario", "")
        context = dict(time=time_, scenario=scenario, region=region, carrier=carrier, tech=tech, subtech=subtech)
        if carrier == "":
            print(f"'carrier' is not defined for processor {p_sim.attrs}, ignored")
            continue  # Ignore processors not having carrier defined
        from enbios import subtech_supported
        if subtech_supported and subtech != g_default_subtech:
            # TODO Add subtech support
            print(f"'subtechnology' field is not really supported, processor {p_sim.attrs} ignored")
            continue  # Ignore processors having subtechnology defined

        # Find "LCI" and "Reference Tech" matching Processor(s)
        # (considering "carrier" - "EcoinventCarrierName" if it is defined)
        # matching_lci, matching_not_accounted_reference_tech = _find_lci_and_tech_processors(prd,
        #                                                                                     structural_not_accounted_procs,
        #                                                                                     dendrogram_musiasem_procs,
        #                                                                                     p_sim)
        matching_not_accounted_reference_techs = _find_lci_and_tech_processors(prd,
                                                                               structural_not_accounted_procs,
                                                                               dendrogram_musiasem_procs,
                                                                               p_sim)
        if len(matching_not_accounted_reference_techs) == 0:
            print(f"Could not find a tech processor matching {p_sim.attrs}. Skipping")
            continue

        for matching_not_accounted_reference_tech, matching_lci in matching_not_accounted_reference_techs.items():
            # Parent processor(s) in "MUSIASEM Dendrogram"
            musiasem_matches = _find_parents(prd, structural_not_accounted_procs, matching_not_accounted_reference_tech)
            if musiasem_matches is None or len(musiasem_matches) == 0:
                print(f"{p_sim} has no parents. Cannot account it, skipped.")
                continue

            iamc_codes_techs[tech] = matching_not_accounted_reference_tech.attributes.get(
                "iamccode", matching_not_accounted_reference_tech.full_hierarchy_name.replace(".", "|"))

            # Add PROCESSOR(S)
            name, parent = _add_processor(p_sim, musiasem_matches, processors, already_added_processors, techs_used_in_regions, context)

            # Add INTERFACES (FROM Simulation AND FROM LCI), if we had a processor added (a name)
            if name:  # At least a match? -> add Interfaces, combining
                _add_interfaces(p_sim, name, parent, matching_lci, matching_not_accounted_reference_tech, context)

    print(f"Generating NIS file ...")  # for: {fragment_metadata}")

    # GENERATE NIS file
    nis_file_name = _generate_fragment_nis_file(clone_processors, cloners, processors, interfaces,
                                                variables, fragment_metadata["scenarios"])

    # SUBMIT NIS file: account and calculate indicators, if a NIS file was generated
    outputs = {}
    if nis_file_name:
        print(f'frag_file = "{nis_file_name}"')
        try:
            print(f"Processing NIS file ...")  # for: {fragment_metadata}")
            _ = ["flow_graph_solution_indicators"]
            if generate_interface_results:
                _.append("flow_graph_solution")
            if generate_nis_fragment_file:
                _.append("model")
            _.append("flow_graph_global_indicators")

            issues, new_state, ds = run_nis_for_results(nis_file_name, development_nis_file, state, _)
            if ds:
                outputs["indicators"] = ds[0]
                outputs["iamc_indicators"] = _prepare_iamc_dataframe_fragment(state, ds[0], iamc_codes_techs, fragment_metadata)
                print(f"File processing done, shape of 'Indicators' dataset: {ds[0].shape}, "
                      f"shape of IAMC dataset: {outputs['iamc_indicators'].shape}")
                if generate_interface_results:
                    outputs["interfaces"] = ds[1]
                if generate_nis_fragment_file:
                    outputs["idempotent_nis"] = ds[2] if len(ds) == 3 else ds[1]
                outputs["global_indicators"] = ds[-1]
            else:
                print(f"There were issues processing fragment file: {nis_file_name}")
                print_issues("Solving fragment", nis_file_name, issues, f"Please check the issues resulting from solving fragment '{nis_file_name}'")

        finally:
            if not keep_fragment_file:
                os.remove(nis_file_name)
    return outputs


if __name__ == '__main__':
    base = "https://docs.google.com/spreadsheets/d/1ZXpxYLVO5BXoxLYeJkvfEiOpYxzNwVy01BmB1hnfNQ0/edit?usp=sharing"  # "Copy of Enviro-Sentinel-Base for Script Development"
    base = "https://docs.google.com/spreadsheets/d/15NNoP8VjC2jlhktT0A8Y0ljqOoTzgar8l42E5-IRD90/edit?usp=sharing"  # Base full
    base = "https://docs.google.com/spreadsheets/d/1AXCBJZEr8Gw6c9OeCTId3TwUaWZt_ePAOx34HgDJdSI/edit?usp=sharing"  # Base full, LCIA implementation
    # base = "https://docs.google.com/spreadsheets/d/1nYzphq1XYrezquW3yj7UiJ8sNJ8RzJ8gqErg3oVmw2Q/edit?usp=sharing"  # Base with only 1 processor
    frag_file = ""
    if os.path.exists(frag_file):
        # state = deserialize_state(prepare_base_state(base, solve=False))
        state = None
        file, df1, df2 = run_nis_for_results(frag_file, None, state)
        df1.to_csv("~/Downloads/df1.csv", index=False)
        df2.to_csv("~/Downloads/df2.csv", index=False)
    else:
        t = Enviro()
        _ = dict(nis_file_location=base,
                 correspondence_files_path="",
                 simulation_type="sentinel",
                 simulation_files_path="/home/rnebot/Downloads/borrame/calliope-output/datapackage.json",
                 output_directory="/home/rnebot/Downloads/borrame/enviro-output-2/")
        t.set_cfg_file_path(_)
        t.compute_indicators_from_base_and_simulation(n_fragments=True,
                                                      generate_nis_base_file=False,
                                                      generate_nis_fragment_file=False,
                                                      generate_interface_results=True,
                                                      generate_indicators=True,
                                                      max_lci_interfaces=0,
                                                      n_cpus=1)
    print("Done")
