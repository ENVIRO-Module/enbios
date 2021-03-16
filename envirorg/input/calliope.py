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
import yaml


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