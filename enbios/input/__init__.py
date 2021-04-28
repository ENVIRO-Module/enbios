class Simulation(ABC):
    """ Abstract simulation reader (the implementation will depend on the type of simulator used) """
    def read(self, simulation_files_path: str):
        """ Read"""
        pass

    def blocks(self):
        """ An iterator into the blocks (technologies) in the simulation """
        pass

    def block_information(self, block_name):
        """ Return information for block """
        pass

