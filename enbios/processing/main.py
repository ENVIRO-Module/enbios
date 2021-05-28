import functools
import logging
import operator
import os
import time
from multiprocessing import Pool, cpu_count
import itertools
import tempfile
import pandas as pd
from typing import Tuple
from NamedAtomicLock import NamedAtomicLock

from nexinfosys.common.decorators import deprecated
from nexinfosys.common.helper import PartialRetrievalDictionary, any_error_issue
from nexinfosys.embedded_nis import NIS
from nexinfosys.model_services import State
from nexinfosys.models.musiasem_concepts import Processor
from nexinfosys.serialization import deserialize_state

from enbios.common.helper import generate_workbook, prepare_base_state, list_to_dataframe, get_valid_name
from enbios.input import Simulation
from enbios.input.lci import LCIIndex
# from enbios.input.simulators.calliope import CalliopeSimulation
from enbios.input.simulators.sentinel import SentinelSimulation
from enbios.processing import read_parse_configuration, read_submit_solve_nis_file
from enbios.processing.model_merger import Matcher, merge_models


#####################################################
# MAIN ENTRY POINT  #################################
#####################################################
_idx_cols = ["region", "scenario", "technology", "model", "carrier",
             "year", "timestep", "unit", "variable", "description"]


def parallelizable_process_fragment(param,
                                    s_state,
                                    tmp_out_dir,
                                    output_dir,
                                    generate_nis_fragment_file,
                                    generate_interface_results,
                                    generate_indicators,
                                    max_lci_interfaces
                                    ):

    def write_outputs(nis_idempotent_file, df_indicators, df_interfaces):
        print("Writing results... --")
        lock = NamedAtomicLock("enbios-lock")
        lock.acquire()
        try:
            indicators_csv_file = output_dir + os.sep + f"indicators.csv"
            if not os.path.isfile(indicators_csv_file):
                df_indicators.to_csv(indicators_csv_file, index=False)
            else:
                df_indicators.to_csv(indicators_csv_file, index=False, mode='a', header=False)
        finally:
            lock.release()

        if generate_nis_fragment_file:
            with open(output_dir + os.sep + f"fragment_nis{get_valid_name(str(p_key))}.xlsx", "wb") as f:
                f.write(nis_idempotent_file)

        if generate_interface_results:
            csv_name = output_dir + os.sep + f"fragment_interfaces{get_valid_name(str(p_key))}.csv"
            if not df_interfaces.empty:
                df_interfaces.to_csv(csv_name, index=False)
            else:
                with open(csv_name, "wt") as f:
                    f.write("Could not obtain interface values (??)")

        if generate_indicators:
            csv_name = output_dir + os.sep + f"fragment_indicators{get_valid_name(str(p_key))}.csv"
            if not df_indicators.empty:
                df_indicators.to_csv(csv_name, index=False)
            else:
                with open(csv_name, "wt") as f:
                    f.write("Could not obtain indicator values (??)")
        print("Writing done. ----- ")

    frag_label, p_key, f_metadata, f_processors = param
    print(f"Fragment processing...")
    start = time.time()

    nis_idempotent_file, df_indicators, df_interfaces = process_fragment(s_state, p_key,
                                                                         f_metadata, f_processors,
                                                                         tmp_out_dir,
                                                                         max_lci_interfaces)

    end = time.time()
    print(f"Fragment processed in {end - start} seconds ---------------------------------------")
    write_outputs(nis_idempotent_file, df_indicators, df_interfaces)
    start = time.time()
    print(f"Fragment outputs written in {start-end} seconds")


class Enviro:
    def __init__(self):
        self._cfg_file_path = None
        self._cfg = None

    def set_cfg_file_path(self, cfg_file_path):
        self._cfg = read_parse_configuration(cfg_file_path)
        self._cfg_file_path = cfg_file_path if isinstance(cfg_file_path, str) else None

    def _prepare_base(self, solve: bool):
        """

        :return:
        """
        return prepare_base_state(self._cfg["nis_file_location"], solve)

    def _get_simulation(self) -> Simulation:
        # Simulation
        simulation = None
        if self._cfg["simulation_type"].lower() == "sentinel":
            simulation = SentinelSimulation(self._cfg["simulation_files_path"])
        # elif self._cfg["simulation_type"].lower() == "calliope":
        #     simulation = CalliopeSimulation(self._cfg["simulation_files_path"])
        return simulation

    def _prepare_process(self) -> Tuple[NIS, LCIIndex, Simulation]:
        # Simulation
        simulation = self._get_simulation()
        # MuSIASEM (NIS)
        nis = read_submit_solve_nis_file(self._cfg["nis_file_location"], state=None, solve=False)
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

    def _read_simulation_fragments(self, split_by_region=True, split_by_scenario=True, split_by_period=True):

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
        # col_types: list of value (not index) fields
        # ct: list of pairs (tech, region)
        prd, scenarios, regions, times, techs, carriers, col_types, ct = simulation.read("")

        partition_lists = []
        if split_by_region and len(regions) > 0:
            partition_lists.append([("_g", r) for r in regions])
        if split_by_period and len(times) > 0:
            partition_lists.append([("_d", t) for t in times])
        if split_by_scenario and len(scenarios) > 0:
            partition_lists.append([("_s", s) for s in scenarios])

        for i, partition in enumerate(list(itertools.product(*partition_lists))):
            partial_key = {t[0]: t[1] for t in partition}
            procs = prd.get(partial_key)
            # Sweep "procs" and update periods, regions, scenarios, models and carriers
            md = dict(periods=set(), regions=set(), scenarios=set(), models=set(), carriers=set(), techs=set(), flows=set())
            for p in procs:
                md["regions"].add(p.attrs["region"])
                md["scenarios"].add(p.attrs["scenario"])
                md["techs"].add(p.attrs["technology"])
                if "year" in p.attrs:
                    md["periods"].add(p.attrs["year"])
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
            yield ':'.join([f"{k}{v}" for k, v in partial_key.items()]), partial_key, md, procs

    def compute_indicators_from_base_and_simulation(self,
                                                    just_one_fragment: bool = False,
                                                    generate_nis_base_file: bool = False,
                                                    generate_nis_fragment_file: bool = False,
                                                    generate_interface_results: bool = False,
                                                    generate_indicators: bool = False,
                                                    max_lci_interfaces: int = 0,
                                                    n_cpus: int = 1):
        """
        MAIN entry point of current ENVIRO
        Previously, a Base NIS must have been prepared, see @_prepare_base

        :param just_one_fragment: True if only one of the fragments is to be computed, to test things
        :param generate_nis_base_file: True if the Base file should be generated (once) for testing purposes
        :param generate_nis_fragment_file: True if the current fragment should be dumped into a NIS formatted XLSX file
        :param generate_interface_results: True if a CSV with values at interfaces should be produced, for each fragment
        :param generate_indicators: True if a CSV with indicators should be produced, for each fragment
        :param max_lci_interfaces: Max number of LCI interfaces to consider. 0 for all (default 0)
        :param n_cpus: Number of CPUs of the local computer used to perform the process
        :return:
        """

        # Prepare Base
        serial_state = self._prepare_base(solve=False)
        output_dir = self._cfg["output_directory"]
        os.makedirs(output_dir, exist_ok=True)
        if generate_nis_base_file:
            nis_file, _, _ = run_nis_for_indicators(None, deserialize_state(serial_state))
            with open(output_dir+os.sep+"nis_base.idempotent.xlsx", "wb") as f:
                f.write(nis_file)

        tmp_output_dir = tempfile.gettempdir()

        # MAIN LOOP - Split simulation in totally independent fragments, and process it
        fragments = sorted([_ for _ in self._read_simulation_fragments()], key=operator.itemgetter(0))
        logging.debug(f"Simulation read, and split in {len(fragments)} fragments")
        if n_cpus == 1:
            for i, (frag_label, partial_key, frag_metadata, frag_processors) in enumerate(fragments):
                # fragment_metadata: dict with regions, years, scenarios in the fragment
                # fragment_processors: list of processors with their attributes which will be interfaces
                print(f"{partial_key}: {len(frag_processors)}")
                parallelizable_process_fragment((frag_label, partial_key, frag_metadata, frag_processors), serial_state,
                                                tmp_output_dir, output_dir,
                                                generate_nis_fragment_file, generate_interface_results,
                                                generate_indicators, max_lci_interfaces)

                if just_one_fragment:
                    break
        else:
            # If 0 -> find appropriate number of CPUs to use
            if n_cpus == 0:
                n_cpus = int(0.8*cpu_count()) if cpu_count() > 4 else 2

            p = Pool(n_cpus)
            p.map(functools.partial(parallelizable_process_fragment,
                                    s_state=serial_state,
                                    tmp_out_dir=tmp_output_dir, output_dir=output_dir,
                                    generate_nis_fragment_file=generate_nis_fragment_file,
                                    generate_interface_results=generate_interface_results,
                                    generate_indicators=generate_indicators,
                                    max_lci_interfaces=max_lci_interfaces),
                  fragments)

    @deprecated  # Use "compute_indicators_from_base_and_simulation"
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
        s = generate_workbook(lst)
        if s:
            with open(temp_name, "wb") as f:
                f.write(s)
        else:
            print(f"ACHTUNG BITTE!: it was not possible to produce XLSX")

        # Execute the NIS file
        nis, issues = read_submit_solve_nis_file(temp_name)

        # TODO Download outputs
        # TODO Elaborate indicators
        # TODO Write indicator files

        # os.remove(temp_name)


def run_nis_for_indicators(nis_file_name, state):
    nis = NIS()
    nis.open_session(True, state)
    if nis_file_name:
        nis.load_workbook(f"file://{nis_file_name}")
    elif isinstance(state, State):
        nis.append_command("Ignore me", list_to_dataframe([["dummy"], ["dummy"]]))
    issues = nis.submit_and_solve()
    tmp = nis.query_available_datasets()
    if not any_error_issue(issues):
        error = False
        outputs = nis.get_results([
            ("model", "Model.xlsx"),
            ("dataset", "flow_graph_solution_indicators", "csv"),
            ("dataset", "flow_graph_solution", "csv"),
        ])
    else:
        error = True
        outputs = nis.get_results([
            ("model", "Model.xlsx")
        ])
    if not error:
        df_indicators = outputs[1][0] if outputs[1][2] else pd.DataFrame()
        df_interfaces = outputs[2][0] if outputs[2][2] else pd.DataFrame()
    else:
        df_indicators = pd.DataFrame()
        df_interfaces = pd.DataFrame()
    nis_file = outputs[0][0]
    # with open(nis_file_name+".idem.xlsx", "wb") as f:
    #     f.write(outputs[0][0])

    # TODO Obtain indicators matrix:
    #   (scenario, processor, region, time, ¿carrier?) -> (i1, ..., in)
    # TODO Append to global indicators matrix (this could be done sending results and another process
    #  would be in charge of assembling)
    nis.close_session()
    return nis_file, df_indicators, df_interfaces


def process_fragment(base_serial_state, partial_key, fragment_metadata, fragment_processors, output_directory,
                     max_lci_interfaces):
    """
        Different assembly options
         * Dendrogram functional processors: a clone per region / a fragment per region (so no need to clone)
        TODO * Structural processors: currently there is a list of structural processors attached to the last level of
         functional processors, with a .spold file associated with them.
         When a simulation processor is found, associate it with one of these processors ("correspondence" file)
         * If there is a .spold file associated to the simulation processor, use it
         * If there is no .spold file and the "parent processor" has one:
           - attach the simulation processor to the parent processor
           - use the

    * if spold(Sim) is None
        if spold(P(Sim)) -> spold(Sim) = spold(P(Sim))
        else ERR
      else OK
      INSERT Sim interfaces
      output_interface_name <- name of the output interface from interfaces in the simulation processor
      for spold_proc in spold(Sim)
        for each interface in spold_proc
          INSERT interface, make them relative_to "output_interface_name"
      COPY all interfaces of spold(Sim) to Sim (insert "interfaces") and make them relative_to the output interface

    * if skip_aux_structural AND is_structural(P(Sim)) AND is_not_accounted(P(Sim)) -> P(Sim) = P(P(Sim))
    """

    def _get_processors_by_type():
        procs = prd.get(Processor.partial_key())
        base_musiasem_procs = {}
        structural_lci_procs = {}
        for proc in procs:
            n_interfaces = len(proc.factors)
            accounted = proc.instance_or_archetype.lower() == "instance"
            functional = proc.functional_or_structural.lower() == "functional"
            if accounted and n_interfaces == 0:
                for hname in proc.full_hierarchy_names(prd):
                    base_musiasem_procs[hname] = proc
            elif not functional and not accounted:
                for hname in proc.full_hierarchy_names(prd):
                    structural_lci_procs[hname] = proc
            else:
                print(f"{proc.full_hierarchy_names(prd)[0]} is neither MuSIASEM base nor LCI base "
                      f"(functional: {functional}; accounted: {accounted}; n_interfaces: {n_interfaces}.")
        return base_musiasem_procs, structural_lci_procs

    def _find_parents(reg: PartialRetrievalDictionary, base_procs, p):
        """
        Find names of parent Processors of "p"

        :param reg: Registry of NIS objects
        :param base_procs: Dict of Processor in the dendrogram
        :param p: name of a simulation process, which should be enumerated in the previous Dict
        :return: Dict of parents, {processor_name: Processor}. None if "p" is not found
        """
        # Find the Processor object
        target = None  # type: Processor
        tech = p.attrs["technology"].lower()
        for proc_name, proc in base_procs.items():
            last = proc_name.split(".")[-1]
            if tech == last.lower():
                target = proc
                break
        if target:
            _ = []
            for n in target.full_hierarchy_names(reg):
                s1 = n.rsplit(".", 1)
                if len(s1) > 1:
                    _.append(s1[0])

            return {n: reg.get(Processor.partial_key(name=n))[0] for n in _}
        else:
            return None  # Not found, cannot tell if it has parents

    def _find_lci(reg: PartialRetrievalDictionary, lci_procs, base_procs, p):
        """
        Find LCI Processor(s) which are assimilated to "p"
          - Processor "p" should exist, as Structural/not-Accounted, with a Parent, [with no Interfaces?]
          - Ecospold file associated
          - "p" does not have an Ecospold, take parent's Ecospold
          - Find a processor (Structural/not-Accounted but also without Parent) with the same Ecospold

        :param reg: Registry of NIS objects
        :param lci_procs: Dict of LCI Processors
        :param base_procs: Dict of Dendrogram Processors
        :param p: Name of the target processor
        :return: List of matching LCI processors, None if "p" is not found
        """
        def _find_pure_lci_processor(target, lci_procs):
            lci_proc = None
            ecospold_file = target.attributes.get("EcoinventFilename", "")
            if ecospold_file != "":
                for proc_name, proc in lci_procs.items():
                    last = proc_name
                    ecospold_file_2 = proc.attributes.get("EcoinventFilename", "")
                    if ecospold_file == ecospold_file_2 and "." not in last:
                        lci_proc = proc
                        break
            return lci_proc

        target = None  # type: Processor
        tech = p.attrs["technology"].lower()
        for proc_name, proc in lci_procs.items():
            last = proc_name.split(".")[-1]
            if tech == last.lower():
                target = proc
                break
        if target:
            lci_proc = _find_pure_lci_processor(target, lci_procs)
            if lci_proc is None:
                # Parent
                s1 = target.full_hierarchy_names(reg)[0].rsplit(".", 1)[0]
                for proc_name, proc in base_procs.items():
                    if s1 == proc_name:
                        lci_proc = _find_pure_lci_processor(proc, lci_procs)
                        break
        else:
            lci_proc = None

        return [lci_proc]

    def _generate_fragment_nis_file(clone_processors, cloners, processors, interfaces, variables, scenarios):
        cmds = []
        # ProblemStatement
        lst = [["Scenario", "Parameter", "Value", "Description"]]
        for s in scenarios:
            # Scenario, Parameter, Value, Description
            lst.append([f"s{s}", "NISSolverObserversPriority", f"o{s}", f"Scenario s{s}, observer o{s}"])
        cmds.append(("ProblemStatement sim fragment", list_to_dataframe(lst)))
        # RefProvenance
        lst = [["RefID", "ProvenanceFileURL", "AgentType", "Agent", "Activities", "Entities"]]
        for s in scenarios:
            lst.append([f"o{s}", "", "Software", f"Sentinel simulation observer for scenario {s}", "WP4", ""])
        cmds.append(("RefProvenance sim fragment", list_to_dataframe(lst)))
        # InterfaceTypes
        lst = [["InterfaceTypeHierarchy", "InterfaceType", "Sphere", "RoegenType", "ParentInterfaceType", "Formula",
                "Description", "Unit", "OppositeSubsystemType", "Attributes"]]
        for v_name, attrs in variables.items():
            lst.append(["sentinel", v_name, "Technosphere", "Flow", "", "",
                        "It is the main output" if attrs["main_flow"] else "", attrs["unit"], "", ""])
        cmds.append(("InterfaceTypes sim fragment", list_to_dataframe(lst)))

        if clone:
            cmds.append(("BareProcessors regions", list_to_dataframe(clone_processors)))
            cmds.append(("ProcessorScalings", list_to_dataframe(cloners)))
        cmds.append(("BareProcessors sim fragment", list_to_dataframe(processors)))
        cmds.append(("Interfaces sim fragment", list_to_dataframe(interfaces)))
        s = generate_workbook(cmds)
        if s:
            temp_name = tempfile.NamedTemporaryFile(dir=output_directory, delete=False)
            with open(temp_name.name, "wb") as f:
                f.write(s)
            return temp_name.name
        else:
            return None

    def is_main_flow(flow_name):  # TODO Depends on simulation
        return flow_name in ["flow_out"]

    def get_flow_orientation(flow_name):  # TODO Depends on simulation
        return "Output" if flow_name in ["flow_out"] else "Input"

    def get_flow_unit(flow_name):  # TODO Depends on simulation
        if flow_name.lower() in ["flow_out", "flow_in"]:
            return "kWh"
        elif flow_name.lower() in ["capacity_factor"]:
            return "kW"
        else:
            return "dimensionless"

    def _interface_used_in_some_indicator(iface):
        return True

    print(f"Processing fragment: {fragment_metadata}")
    state = deserialize_state(base_serial_state)
    prd = state.get("_glb_idx")
    base_musiasem_procs, structural_lci_procs = _get_processors_by_type()

    processors = [
        ["ProcessorGroup", "Processor", "ParentProcessor", "SubsystemType", "System", "FunctionalOrStructural",
         "Accounted", "Stock", "Description", "GeolocationRef", "GeolocationCode", "GeolocationLatLong", "Attributes",
         "@SimulationName", "@Region"]]  # TODO Maybe more fields to qualify the fragment
    interfaces = [
        ["Processor", "InterfaceType", "Interface", "Sphere", "RoegenType", "Orientation", "OppositeSubsystemType",
         "GeolocationRef", "GeolocationCode", "InterfaceAttributes", "Value", "Unit", "RelativeTo", "Uncertainty",
         "Assessment", "PedigreeMatrix", "Pedigree", "Time", "Source", "NumberAttributes", "Comments"]]
    clone = False
    if clone:
        cloners = [
            ["InvokingProcessor", "RequestedProcessor", "ScalingType", "InvokingInterface", "RequestedInterface",
             "Scale"]]
        clone_processors = [
            ["ProcessorGroup", "Processor", "ParentProcessor", "SubsystemType", "System", "FunctionalOrStructural",
             "Accounted", "Stock", "Description", "GeolocationRef", "GeolocationCode", "GeolocationLatLong",
             "Attributes",
             "@SimulationName", "@Region"]]
        # For each region, create a processor in "clone_processors" and an entry in "cloners" hanging the top
        # Processors in the dendrogram into the region processor
        for r in fragment_metadata["regions"]:
            name = None
            clone_processors.append(["", r, "", "", r, "Functional", "Yes", "",
                                     f"Country level processor, region '{r}'", "", "", "", "", r, r])
            considered = set()
            for p_name, p in base_musiasem_procs.items():
                if len(p_name.split(".")) > 1 and not p in considered:
                    considered.add(p)
                    # InvokingInterface, RequestedInterface, Scale are mandatory; specify something here
                    cloners.append((r, p_name, "Clone", "", "", ""))
    else:
        cloners = []
        clone_processors = []
        # Also, modify all parent processors system to "r"
        if len(fragment_metadata["regions"]) == 1:
            r = next(iter(fragment_metadata["regions"]))
            for p in base_musiasem_procs.values():
                p.processor_system = r

    already_added_processors = set()
    already_added_lci = set()
    variables = dict()
    max_lci_interfaces = 100000 if max_lci_interfaces == 0 else max_lci_interfaces
    # If not periods are defined, define a specific year. At least one is needed for the solver to run
    default_time = "2038" if len(fragment_metadata["periods"]) == 0 else "Year"
    for p in fragment_processors:  # Each Simulation processor
        # Update time and scenario (they can change from entry to entry, but for MuSIASEM it is the same entity)
        time_ = p.attrs.get("year", default_time)
        scenario = p.attrs.get("scenario", "")
        region = p.attrs.get("region", "")
        # Find Dendrogram matches
        musiasem_matches = _find_parents(prd, structural_lci_procs, p)
        if musiasem_matches is None or len(musiasem_matches) == 0:
            print(f"{p} has no parents. Cannot account it, skipped.")
            continue
        first_name = None
        for m_name, m_proc in musiasem_matches.items():
            name = f"{p.attrs['technology']}_{region}"  # TODO Maybe "carrier"? depends on the intention
            parent = m_name
            description = f"Simulation {p.attrs['technology']} for region {region}, scenario {scenario}."
            original_name = f"{p.attrs['technology']}"
            if first_name is None:
                first_name = f"{parent}.{name}"
                if (name, parent) not in already_added_processors:
                    processors.append(["", name, parent, "", region, "Structural", "Yes", "",
                                       description, "", "", "", "", original_name, region])
                    already_added_processors.add((name, parent))
            else:
                if (name, parent) not in already_added_processors:
                    processors.append(["", first_name, parent, "", region, "Structural", "Yes", "",
                                       description, "", "", "", "", original_name, region])
                    already_added_processors.add((name, parent))

        if first_name is not None:
            relative_to = None
            for i in set(p.attrs.keys()).difference(_idx_cols):
                if is_main_flow(i):
                    relative_to = i
                v = p.attrs[i]
                orientation = get_flow_orientation(i)
                interfaces.append([name, i, "", "", "", orientation, "", "", "", "", v,
                                   "", "", "", "", "", "", time_, f"o{scenario}", "", ""])
                if i not in variables:
                    variables[i] = dict(main_flow=is_main_flow(i), orientation=orientation, unit=get_flow_unit(i))

            if (name, parent) not in already_added_lci:
                already_added_lci.add((name, parent))
                # Find LCI matches, expand interfaces
                lci_matches = _find_lci(prd, structural_lci_procs, base_musiasem_procs, p)
                if lci_matches is None or len(lci_matches) == 0:
                    print(f"{p} has no LCI data associated. Cannot provide information for indicators, skipped.")
                    continue
                for cp in lci_matches:
                    for cont, i in enumerate(cp.factors):
                        if cont >= max_lci_interfaces:
                            break  # Artificial limit to reduce file size, for test purposes

                        if not _interface_used_in_some_indicator(i.name):  # Economy of the model: avoid specifying interfaces not used later
                            continue
                        q = i.quantitative_observations[0]
                        if q.attributes.get("relative_to", None) is None:
                            continue
                        unit = q.attributes.get("unit")  # TODO unit / relative_to_unit
                        orientation = i.attributes.get("orientation", "")
                        v = q.value
                        # TODO Static: does not depend on scenario or time, avoid inserting repeatedly
                        interfaces.append([name, i.name, "", "", "", orientation, "", "", "", "", v,
                                           "", relative_to, "", "", "", "", "Year", "", "", ""])

    print(f"Generating NIS file for: {fragment_metadata}")

    # Generate and run NIS file, returning the result
    nis_file_name = _generate_fragment_nis_file(clone_processors, cloners, processors, interfaces,
                                                variables, fragment_metadata["scenarios"])
    if nis_file_name:
        print(f'frag_file = "{nis_file_name}"')
        # sys.exit(1)  # For now
        try:
            print(f"Processing NIS file for: {fragment_metadata}")
            nis_idempotent, df_indicators, df_interfaces = run_nis_for_indicators(nis_file_name, state)
            print(f"Processing NIS file done, result shape: {df_indicators.shape}")
        finally:
            os.remove(nis_file_name)
        return nis_idempotent, df_indicators, df_interfaces
    else:
        return None, pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    base = "https://docs.google.com/spreadsheets/d/1ZXpxYLVO5BXoxLYeJkvfEiOpYxzNwVy01BmB1hnfNQ0/edit?usp=sharing"  # "Copy of Enviro-Sentinel-Base for Script Development"
    base = "https://docs.google.com/spreadsheets/d/15NNoP8VjC2jlhktT0A8Y0ljqOoTzgar8l42E5-IRD90/edit?usp=sharing"  # Base full
    # base = "https://docs.google.com/spreadsheets/d/1nYzphq1XYrezquW3yj7UiJ8sNJ8RzJ8gqErg3oVmw2Q/edit?usp=sharing"  # Base with only 1 processor
    frag_file = "/tmp/tmp2huk3hdy"  # "/tmp/tmpp4shvdiw"
    frag_file = "/home/rnebot/Downloads/simple_relative_to.xlsx"
    frag_file = ""
    if os.path.exists(frag_file):
        # state = deserialize_state(prepare_base_state(base, solve=False))
        state = None
        file, df1, df2 = run_nis_for_indicators(frag_file, state)
        df1.to_csv("~/Downloads/df1.csv", index=False)
        df2.to_csv("~/Downloads/df2.csv", index=False)
    else:
        t = Enviro()
        _ = dict(nis_file_location=base,
                 correspondence_files_path="",
                 simulation_type="sentinel",
                 simulation_files_path="/home/rnebot/Downloads/borrame/calliope-output/datapackage.json",
                 lci_data_locations={},
                 output_directory="/home/rnebot/Downloads/borrame/enviro-output/")
        t.set_cfg_file_path(_)
        t.compute_indicators_from_base_and_simulation(just_one_fragment=True,
                                                      generate_nis_base_file=True,
                                                      generate_nis_fragment_file=True,
                                                      generate_interface_results=True,
                                                      generate_indicators=True,
                                                      max_lci_interfaces=0,
                                                      n_cpus=1)
    print("Done")
