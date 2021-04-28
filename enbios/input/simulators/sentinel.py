"""

Read a Sentinel model definition and outputs, to obtain one or more DiGraphs which can be used to
obtain either a MuSIASEM or LCA structure

"""
from typing import Dict

from enbios.input import Simulation


class SentinelSimulation(Simulation):
    def __init__(self, sentinel_index_path):
        self._sentinel_index = sentinel_index_path

    def read(self, simulation_files_path: str):
        """ Read variables and datasets """
        pass

    def blocks(self):
        """ An iterator into the blocks (technologies) in the simulation """
        pass

    def block_information(self, block_name) -> Dict[str, str]:
        """ Return information for block. A dictionary """
        pass


dp = "/home/rnebot/Downloads/borrame/calliope-output"
