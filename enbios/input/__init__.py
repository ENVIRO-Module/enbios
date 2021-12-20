from abc import ABC


class Simulation(ABC):
    """ Abstract simulation reader (the implementation will depend on the type of simulator used) """
    def read(self, filter_model: str, default_time=None):
        """ Read"""
        pass

    def blocks(self):
        """ An iterator into the blocks (technologies) in the simulation """
        pass

    def block_information(self, block_name):
        """ Return information for block """
        pass


