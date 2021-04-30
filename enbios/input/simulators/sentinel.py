"""

Read a Sentinel model definition and outputs, to obtain one or more DiGraphs which can be used to
obtain either a MuSIASEM or LCA structure

"""
from typing import Dict

from enbios.input import Simulation
from friendly_data.dpkg import read_pkg
from friendly_data.converters import to_df


class SentinelSimulation(Simulation):
    def __init__(self, sentinel_index_path):
        self._sentinel_index_path = sentinel_index_path
        self._blocks = {}
        self.read()

    def read(self):
        """ Read variables and datasets """
        pkg = read_pkg(self._sentinel_index_path)
        lst = []
        for r in pkg.resources:
            df = to_df(r)
            lst.append(df)
        print(pkg)

    def blocks(self):
        """ An iterator into the blocks (technologies) in the simulation """
        pass

    def block_information(self, block_name) -> Dict[str, str]:
        """ Return information for block. A dictionary """
        pass


dp = "/home/rnebot/Downloads/borrame/calliope-output"
