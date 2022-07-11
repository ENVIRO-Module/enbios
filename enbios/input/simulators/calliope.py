"""
Read both Calliope model and Calliope outputs

* Model into Graph
* Outputs into same Graph
  - dispatch decisions. Useful to evaluate consumption of resource. But compute only accumulated consumption instead of the Time Series
  - capacities. Useful to scale processors/activities


MODEL -> RUN -> ASSESS (read model and results; match model to data; elaborate NIS and BW input; run NIS and BW; get results)


"""

import networkx as nx
import calliope
from typing import List

from enbios.input import Simulation


def read_calliope_model(model_file: str, outputs: str) -> List[nx.DiGraph]:
    """
    Read a Calliope model and return a graph of technologies/sites per scenario
    Also return Calliope model outputs and integrate them into the Graph

    :param base_path:
    :return:
    """
    # Read locations, techs, link-techs, and scenarios (different models)
    # "import" enumerates other files to import, recursively
    model = calliope.Model(model_file)


def map_calliope_to_available(calliope: nx.DiGraph, technologies):
    """
    Although the idea is to obtain real technologies instead of generic ones, based on which LCA assessment can be performed,
    maybe there is more than one technology matching the one modeled.
    The matching criteria could be: location, time, technology type, range (large, medium, small instance), HA, surface, price, ...


    :param calliope:
    :param technologies:
    :return:
    """


class CalliopeSimulation(Simulation):
    """
    Abandoned by Calliope during Sentinel project, in favor of Friendly-Data format, see "sentinel.py"
    """
    def __init__(self, simulation_files_path):
        self._model = None
        self._simulation_files_path = simulation_files_path
        self.read()

    def read(self, filter_model: str):
        self._model = calliope.Model(self._simulation_files_path)
        ds = self._model.inputs
        # Carriers -> InterfaceTypes
        carriers = dict()
        for i in ds.carriers:
            s = str(i.values)
            carriers[s] = dict()

        # Locations -> GeographicalReferences
        locations = dict()
        for i in ds.locs:
            s = str(i.values)
            locations[s] = [ds.loc_coordinates.sel(coordinates=gc, locs=s) for gc in ["lat", "lon"]]
        # Technologies -> Processors (Archetype)
        techs = dict()  # TODO: is Transmission, is Conversion, type
        for i in ds.techs:
            s = str(i.values)
            ttype = None
            techs[s] = dict(tech_type=ttype)

        # Technologies at locations -> Processors (Instances) (need to clone Archetypes? seems no)
        # - Type
        # - Efficiency
        # - Land Use
        # - Power Capacity
        techs_locs = dict()
        for i in ds.loc_techs:
            location, tech = str(i.values).split("::")
            techs_locs[f"{location}__{tech}"] = dict(loc=location, tech=tech)
        print("hola")

        # Transmission are technologies -> Processors, ¿allocated to two locations? ¿proportion 50/50?
        links = dict()
        for i in ds.loc_techs_transmission:
            print(i)

        # TODO
        #  * Read locations (and geographical coordinates)
        #  * Read techs (and general features)
        #  * Read pairs location-tech (and particular features)
        #  * Reading features may imply (or not) going through input CSV files

        # TODO f"{tech}_at_{loc}" -> processor: where? lat, long?

        # TODO Can Calliope read the output of the optimization? or is it necessary to read separately...

    def blocks(self):
        """ List of technologies """

    def block_information(self, block_name):
        """ Return information for block """
        pass
