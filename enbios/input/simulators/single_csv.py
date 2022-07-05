from typing import Tuple

import pandas as pd
from nexinfosys.command_generators.parser_ast_evaluators import get_nis_name
from nexinfosys.common.helper import PartialRetrievalDictionary

from enbios.common.helper import list_to_dataframe, get_scenario_name, isfloat
from enbios.input import Simulation
from enbios.input.simulators import create_register_or_update_processor_and_attributes, find_column_idx_name
from enbios.model import SimStructuralProcessorAttributes, g_default_subtech


class AllInACSV(Simulation):
    def __init__(self, file_path):
        self._file_path = file_path

    def read(self, filter_model: str, default_time: str = None):
        """
        Reads CSV file with this format

scenario, region, time, technology, carrier, flow1, flow2
s1, r1, 2020, tech1, carrier1, 3, 2
s1, r1, 2020, tech2, carrier1, 2.5, 1.8
s1, r1, 2020, tech3, carrier1, 1.5, 2.8

        scenario, region, time, technology, carrier are significant (case insensitive)

        :param filter_model: Read only the specified model
        :param default_time: Default time to use if none specified in the inputs
        :return: A tuple with assorted elements, from lists of possible values for dimensions to a registry of Processors
        """

        if filter_model is None:  # Mandatory
            return

        scenarios = set()
        regions = set()
        times = set()
        techs = set()
        carriers = set()
        units = set()
        col_types = set()  # Interface types
        ctc = set()  # Country - Tech - Carrier
        # print(f"SENTINEL INDEX: {self._sentinel_index_path}")
        df = pd.read_csv(self._file_path, sep=None, engine="python")  # sep=None -> "detect separator"
        df.columns = [c.strip().lower() for c in df.columns]
        prd = PartialRetrievalDictionary()
        col_types.update(df.columns)
        region_idx, _ = find_column_idx_name(df.columns, ["region", "regions", "loc", "locs"])
        carrier_idx, _ = find_column_idx_name(df.columns, ["carrier", "carriers"])
        tech_idx, _ = find_column_idx_name(df.columns, ["technology", "technologies", "tech", "techs", "sector", "sectors"])
        subtech_idx, _ = find_column_idx_name(df.columns, ["subtechnology", "subtechnologies", "subtech", "subtechs", "subsector", "subsectors"])
        scenario_idx, _ = find_column_idx_name(df.columns, ["scenario", "scenarios", "storyline", "storylines"])
        subscenario_idx, _ = find_column_idx_name(df.columns, ["subscenario", "subscenarios", "substoryline", "substorylines", "spore", "spores"])
        time_idx, _ = find_column_idx_name(df.columns, ["time", "year", "years"])
        unit_idx, _ = find_column_idx_name(df.columns, ["unit", "units"])
        idxs = set([region_idx, carrier_idx, tech_idx, subtech_idx, scenario_idx, subscenario_idx, time_idx, unit_idx])
        for idx in idxs:
            if idx >= 0:
                col_types.remove(df.columns[idx])
        col_index = {c: df.columns.get_loc(c) for c in col_types}
        for idx_, cols in df.iterrows():
            cols = [c.strip() if not isfloat(c) else c for c in cols]
            region = get_nis_name(cols[region_idx]) if region_idx >= 0 else None
            if carrier_idx >= 0 and isinstance(cols[carrier_idx], str):
                carrier = get_nis_name(cols[carrier_idx])
            else:
                carrier = None
            tech = get_nis_name(cols[tech_idx]) if tech_idx >= 0 else None
            subtech = get_nis_name(cols[subtech_idx]) if subtech_idx >= 0 else None
            scenario = get_nis_name(cols[scenario_idx]) if scenario_idx >= 0 else None
            subscenario = get_nis_name(cols[subscenario_idx]) if subscenario_idx >= 0 else None
            time_ = str(cols[time_idx]) if time_idx >= 0 else default_time
            unit_ = cols[unit_idx] if unit_idx >= 0 else None
            if region:
                regions.add(region)
            if carrier:
                carriers.add(carrier)
            if scenario:
                scenarios.add(scenario)
            if time_:
                times.add(time_)
            if tech:
                if subtech:
                    techs.add(f"{tech}:{subtech}")
                else:
                    techs.add(tech)
            if tech and region:
                if subtech:
                    t = f"{tech}:{subtech}"
                else:
                    t = tech
                ctc.add((t, region, carrier))  # Carrier can be None
            if unit_:
                units.add(unit_)
            if not region and not scenario and not time_:
                region = scenario = time_ = "-"
            # -- Add COLS information --
            d = {col: cols[col_index[col]] for col in col_types}
            create_register_or_update_processor_and_attributes(prd, tech, region, carrier, scenario, time_, subtech, subscenario, carrier_idx, d)
            if subtech:
                # Subtech == g_default_subtech (something different from None)
                create_register_or_update_processor_and_attributes(prd, tech, region, carrier, scenario, time_, g_default_subtech, subscenario, carrier_idx, d)

        return prd, scenarios, regions, times, techs, carriers, units, col_types, ctc

    def generate_template_data(self):
        """
        Produce a set of (tec, country) tuples and a set of (tec, -) tuples
        to help in producing "correspondence"
        :return:
        """
        prd, scenarios, regions, times, techs, carriers, units, col_types, ct = self.read("")

        cmds = []
        # Not NIS
        lst = [["technology", "calliope_technology_type"]]
        for t in techs:
            lst.append([t, ""])
        cmds.append(("Simulation Technology Types", list_to_dataframe(lst)))

        lst = [["region"]]
        for r in regions:
            lst.append([r])
        cmds.append(("Simulation Regions", list_to_dataframe(lst)))

        lst = [["carrier"]]
        for r in carriers:
            lst.append([r])
        cmds.append(("Simulation Carriers", list_to_dataframe(lst)))

        lst = [["period"]]
        for r in times:
            lst.append([r])
        cmds.append(("Simulation Times", list_to_dataframe(lst)))

        lst = [["unit"]]
        for r in units:
            lst.append([r])
        cmds.append(("Simulation Units", list_to_dataframe(lst)))

        lst = [["name", "region", "match_target_type", "match_target", "weight", "match_conditions"]]
        for t in techs:
            lst.append([t, "", "musiasem", "<musiasem parent>", 1, ""])
        for t in techs:
            lst.append([t, "", "lca", "<spold>", 1, ""])
        for t in ct:
            lst.append([t[0], t[1], "musiasem", "<musiasem parent>", 1, ""])
        for t in ct:
            lst.append([t[0], t[1], "lca", "<spold>", 1, ""])
        cmds.append(("Enbios Correspondence", list_to_dataframe(lst)))

        # InterfaceTypes
        lst = [["InterfaceTypeHierarchy", "InterfaceType", "Sphere", "RoegenType", "ParentInterfaceType", "Formula",
                "Description", "Unit", "OppositeSubsystemType", "Attributes"]]
        for col_type in col_types:
            lst.append(["sentinel", col_type, "Technosphere", "Flow", "", "",
                        "<description>", "<unit>", "<opposite>", ""])
        cmds.append(("InterfaceTypes", list_to_dataframe(lst)))

        # # Georeferences
        # geo_references = []
        # for r in regions:
        #     # Georeferences (regions)
        #     print(r)

        # ProblemStatement
        lst = [["Scenario", "Parameter", "Value", "Description"]]
        for s in scenarios:
            # Scenario, Parameter, Value, Description
            scenario = get_scenario_name("s", s)
            observer = get_scenario_name("o", s)
            lst.append([scenario, "NISSolverObserversPriority", observer, f"Scenario {scenario}, observer {observer}"])
        cmds.append(("ProblemStatement", list_to_dataframe(lst)))

        # Processors, Interfaces and "Cloners"
        processors = [["ProcessorGroup", "Processor", "ParentProcessor", "SubsystemType", "System", "FunctionalOrStructural",
                "Accounted", "Stock", "Description", "GeolocationRef", "GeolocationCode", "GeolocationLatLong",
                "Attributes", "@tech_name", "@region"]]
        interfaces = [
            ["Processor", "InterfaceType", "Interface", "Sphere", "RoegenType", "Orientation", "OppositeSubsystemType",
             "GeolocationRef", "GeolocationCode", "InterfaceAttributes", "Value", "Unit", "RelativeTo", "Uncertainty",
             "Assessment", "PedigreeMatrix", "Pedigree", "Time", "Source", "NumberAttributes", "Comments"]]
        cloners = [["InvokingProcessor", "RequestedProcessor", "ScalingType", "InvokingInterface",
                    "RequestedInterface", "Scale"]]
        country_level_procs_already_added = set()
        tech_procs_already_added = set()
        system_per_region = False
        for t in ct:
            o = prd.get(SimStructuralProcessorAttributes.partial_key(t[0], t[1]))
            if len(o) > 0:
                parent = ""  # TODO
                if t[1]:  # Region
                    name = f"ES_{t[1]}"
                    parent = name
                    system = t[1] if system_per_region else ""
                    if name not in country_level_procs_already_added:
                        processors.append(["", name, "", "", system, "Functional", "Yes", "",
                                           f"Country level processor, {t[1]}", "", "", "", "", t[1], t[1]])
                        # Clone (clone dendrogram inside "region" processor)
                        # TODO InvokingInterface, RequestedInterface, Scale are mandatory; specify something here
                        cloners.append((name, "EnergySector", "Clone", "", "", ""))
                        country_level_procs_already_added.add(name)
                else:
                    system = ""
                # BareProcessors
                o1 = o[0]  # prd.get(ProcessorAttributes.partial_key(t[0], t[1]), full_key=True)[0]
                original_name = t[0]
                name = get_nis_name(t[0])
                full_name = f"{parent}{'.' if parent!='' else ''}{name}"
                if name not in tech_procs_already_added:
                    add = True
                else:
                    add = False

                # Technology (structural) processor
                description = f'{o1.attrs.get("description", name)} at {t[1]}'
                processors.append(["", name, parent, "", system, "Structural", "Yes", "",
                                   description, "", "", "", "", original_name, t[1]])
                # Add observations (Interfaces)
                for p in o:
                    observer = ""
                    time_ = "Year"
                    # name = ""
                    region = ""
                    _ = p.attrs.copy()
                    if "scenario" in p.attrs:
                        observer = get_scenario_name("o", p.attrs["scenario"])
                        del _["scenario"]
                    if "technology" in p.attrs:
                        name_ = p.attrs["technology"]
                        del _["technology"]
                    if "region" in p.attrs:
                        region = p.attrs["region"]
                        del _["region"]
                    if "year" in p.attrs or "time" in p.attrs:
                        if "year" in p.attrs:
                            time_ = p.attrs["year"]
                            del _["year"]
                        else:
                            time_ = p.attrs["time"]
                            del _["time"]
                    for k, v in _.items():
                        interfaces.append([full_name, k, "", "Technosphere", "Flow", "<orientation>", "", "", "", "", v,
                                           "", "<relative_to>", "", "", "", "", time_, observer, "", ""])

        cmds.append(("BareProcessors", list_to_dataframe(processors)))
        cmds.append(("ProcessorScalings", list_to_dataframe(cloners)))
        cmds.append(("Interfaces", list_to_dataframe(interfaces)))

        return cmds

    def generate_nis_models(self, split_by_region=False, split_by_scenario=True, split_by_period=True):
        """
        Produce a set of (tec, country) processors with interface values,
        separated by scenario (NIS observer) and year (NIS time)
        :return:
        """

        # Yield NIS commands and a descriptor of which region


if __name__ == '__main__':
    tmp = AllInACSV("/home/rnebot/Downloads/Test_Rafa.csv").read("", "2022")
    tmp = AllInACSV("/home/rnebot/GoogleDrive/AA_SENTINEL/all_in_one.csv").read("", "2022")
