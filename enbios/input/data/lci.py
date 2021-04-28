import os.path
from typing import List

from bw2io.extractors import Ecospold2DataExtractor


class LCIIndex:
    """ An index of where is each activity, to ease reading it """

    def __init__(self, lci_data_locations: List[str]):
        # TODO Prepare an index of activities, and where each is stored. Maybe also the flows
        self._lci_data_locations = lci_data_locations
        self._activity_locations = {}
        self._activities = {}

    def _read(self):
        self._activities = {}
        self._activity_locations = {}
        for ll in self._lci_data_locations:
            data = Ecospold2DataExtractor.extract(ll, "test", use_mp=False)
            self._activities.update({d["activity"]: d["exchanges"] for d in data})
            self._activity_locations({d["activity"]: ll for d in data})

    def activities(self):
        """
        All indexed activities
        :return:
        """
        return self._activity_locations.keys()

    def activity_location(self, activity):
        return self._activity_locations[activity]

    def read_activity(self, activity):
        """
        Read activity as a dictionary so later it can be inserted into an Interfaces command
        (intensive quantities assumed)

        :param activity:
        :return:
        """
        data = Ecospold2DataExtractor.extract(self.activity_location(activity), "test", use_mp=False)
        return [d["exchanges"] for d in data if d["activity"] == activity][0]
