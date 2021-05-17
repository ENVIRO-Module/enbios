"""

Read a Sentinel model definition and outputs, to obtain one or more DiGraphs which can be used to
obtain either a MuSIASEM or LCA structure

"""
from typing import Dict

from nexinfosys.command_generators.parser_ast_evaluators import get_nis_name
from nexinfosys.common.helper import PartialRetrievalDictionary
from enbios.common.helper import list_to_dataframe
from enbios.input import Simulation
from enbios.model import ProcessorAttributes
from friendly_data.dpkg import read_pkg
from friendly_data.converters import to_df


class SentinelSimulation(Simulation):
    def __init__(self, sentinel_index_path):
        self._sentinel_index_path = sentinel_index_path

    def read_raw(self):
        """ Read variables and datasets """
        pkg = read_pkg(self._sentinel_index_path)
        lst = []
        for r in pkg.resources:
            df = to_df(r)
            lst.append(df)

    def read(self, filter_model: str):
        """
        Reads Sentinel package completely

        :param filter_model: Read only the specified model
        :return: A tuple with assorted elements, from lists of possible values for dimensions to a registry of Processors
        """
        if filter_model is None:  # Mandatory
            return

        regions = set()
        scenarios = set()
        times = set()
        techs = set()
        carriers = set()
        col_types = set()
        ct = set()
        pkg = read_pkg(self._sentinel_index_path)
        prd = PartialRetrievalDictionary()
        for res in pkg.resources:
            df = to_df(res)
            col_types.update(df.columns)
            region_idx = df.index.names.index("region") if "region" in df.index.names else -1
            carrier_idx = df.index.names.index("carrier") if "carrier" in df.index.names else -1
            tech_idx = df.index.names.index("technology") if "technology" in df.index.names else -1
            scenario_idx = df.index.names.index("scenario") if "scenario" in df.index.names else -1
            time_idx = df.index.names.index("year") if "year" in df.index.names else -1
            for idx, cols in df.iterrows():
                if not isinstance(idx, tuple):  # When it is not a MultiIndex, Pandas has "idx" to not be a Tuple; workaround: convert into a Tuple of a single element
                    idx = (idx,)
                region = idx[region_idx] if region_idx >= 0 else None
                carrier = idx[carrier_idx] if carrier_idx >= 0 else None
                tech = idx[tech_idx] if tech_idx >= 0 else None
                scenario = idx[scenario_idx] if scenario_idx >= 0 else None
                time_ = idx[time_idx] if time_idx >= 0 else None
                if region:
                    regions.add(region)
                if carrier:
                    carriers.add(carrier)
                if scenario:
                    scenarios.add(scenario)
                if time_:
                    times.add(time_)
                if tech:
                    techs.add(tech)
                if tech and region:
                    ct.add((tech, region))
                if not region and not scenario and not time_:
                    region = scenario = time_ = "-"
                # -- Add COLS information --
                # (tech, region, scenario, time) - (cols)
                k = ProcessorAttributes.partial_key(tech, region, scenario, time_)
                o = prd.get(k)
                if len(o) == 0:
                    pa = ProcessorAttributes(tech, region, scenario, time_)
                    prd.put(k, pa)
                elif len(o) == 1:
                    pa = o[0]
                else:
                    raise Exception(f"Found {len(o)} occurrences of ProcessorAttributes: {k}")
                pa.attrs.update({k: cols[k] for k in df.columns})
        return prd, scenarios, regions, times, techs, carriers, col_types, ct

    def generate_template_data(self):
        """
        Produce a set of (tec, country) tuples and a set of (tec, -) tuples
        to help in producing "correspondence"
        :return:
        """
        prd, scenarios, regions, times, techs, carriers, col_types, ct = self.read("")

        cmds = []
        # Not NIS
        lst = [["technology", "calliope_technology_type"]]
        for t in techs:
            lst.append([t, ""])
        cmds.append(("Enbios Technology Types", list_to_dataframe(lst)))

        lst = [["region"]]
        for r in regions:
            lst.append([r])
        cmds.append(("Enbios Regions", list_to_dataframe(lst)))

        lst = [["carrier"]]
        for r in carriers:
            lst.append([r])
        cmds.append(("Enbios Carriers", list_to_dataframe(lst)))

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

        # Georeferences
        geo_references = []
        for r in regions:
            # Georeferences (regions)
            print(r)

        # ProblemStatement
        lst = [["Scenario", "Parameter", "Value", "Description"]]
        for s in scenarios:
            # Scenario, Parameter, Value, Description
            lst.append([f"s{s}", "NISSolverObserversPriority", f"o{s}", f"Scenario s{s}, observer o{s}"])
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
        for t in ct:
            o = prd.get(ProcessorAttributes.partial_key(t[0], t[1]))
            if len(o) > 0:
                if t[1]:  # Region
                    name = f"ES_{t[1]}"
                    processors.append(["<pgroup>", name, "", "", "<system>", "Functional", "Yes", "",
                                       f"Country level processor, {t[1]}", "", "", "", "", t[1], t[1]])
                    # Clone (clone dendrogram inside "region" processor)
                    # TODO InvokingInterface, RequestedInterface, Scale are mandatory; specify something here
                    cloners.append((name, "EnergySector", "Clone", "", "", ""))
                # BareProcessors
                o1 = prd.get_one(ProcessorAttributes.partial_key(t[0], "-", "-", "-"), full_key=True)
                original_name = t[0]
                name = get_nis_name(t[0])
                parent = None  # TODO
                description = f'{o1.attrs["description"]} at {t[1]}'
                # TODO FULL NAME?
                processors.append(["<pgroup>", name, parent, "", "system", "Structural", "Yes", "",
                                   description, "", "", "", "", original_name, t[1]])
                for p in o:
                    scenario = ""
                    time_ = "Year"
                    name = ""
                    region = ""
                    _ = p.attrs.copy()
                    if "scenario" in p.attrs:
                        scenario = "o" + p.attrs["scenario"]
                        del _["scenario"]
                    if "technology" in p.attrs:
                        name_ = p.attrs["technology"]
                        del _["technology"]
                    if "region" in p.attrs:
                        region = p.attrs["region"]
                        del _["region"]
                    if "year" in p.attrs:
                        time_ = p.attrs["year"]
                        del _["year"]
                    for k, v in _.items():
                        interfaces.append([name, k, "", "Technosphere", "Flow", "<orientation>", "", "", "", "", v,
                                           "", "<relative_to>", "", "", "", "", time_, scenario, "", ""])

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


dp = "/home/rnebot/Downloads/borrame/calliope-output"
