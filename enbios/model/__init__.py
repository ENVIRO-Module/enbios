

class SimStructuralProcessorAttributes:
    """
    A class to store Processor attributes while a Sentinel datapackage is read
    and values are stored in a PartialRetrievalDictionary
    Attributes can be stored directly
    """
    def __init__(self, technology=None, region=None, carrier=None, scenario=None, time_=None):
        self.attrs = {}
        if technology:
            self.attrs["technology"] = technology
        if region:
            self.attrs["region"] = region
        if carrier:
            self.attrs["carrier"] = carrier
        if scenario:
            self.attrs["scenario"] = scenario
        if time_:
            self.attrs["time"] = time_

    @staticmethod
    def partial_key(technology=None, region=None, carrier=None, scenario=None, time=None):
        d = {}
        if technology:
            d["_t"] = technology
        if region:
            d["_g"] = region
        if carrier:
            d["_c"] = carrier
        if scenario:
            d["_s"] = scenario
        if time:
            d["_d"] = time
        return d

    def key(self):
        return self.partial_key(self.attrs.get("technology"),
                                self.attrs.get("region"),
                                self.attrs.get("carrier"),
                                self.attrs.get("scenario"),
                                self.attrs.get("time"))

