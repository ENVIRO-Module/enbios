
g_default_subtech = "_"  # Constant for default subtech


class SimStructuralProcessorAttributes:
    """
    A class to store Processor attributes while a Sentinel datapackage is read
    and values are stored in a PartialRetrievalDictionary
    Attributes can be stored directly
    """
    def __init__(self, technology=None, region=None, carrier=None, scenario=None, time_=None,
                 subtechnology=None, subscenario=None):
        self.attrs = {}
        if technology:
            self.attrs["technology"] = technology
        if subtechnology:
            self.attrs["subtechnology"] = subtechnology
        if region:
            self.attrs["region"] = region
        if carrier:
            self.attrs["carrier"] = carrier
        if scenario:
            self.attrs["scenario"] = scenario
        if subscenario:
            self.attrs["subscenario"] = subscenario
        if time_:
            self.attrs["time"] = time_

    @staticmethod
    def partial_key(technology=None, region=None, carrier=None, scenario=None, time=None,
                    subtechnology=None, subscenario=None):
        d = {}
        if technology:
            d["_t"] = technology
        if subtechnology:
            d["_st"] = subtechnology
        if region:
            d["_g"] = region
        if carrier:
            d["_c"] = carrier
        if scenario:
            d["_s"] = scenario
        if subscenario:
            d["_ss"] = subscenario
        if time:
            d["_d"] = time
        return d

    def key(self):
        return self.partial_key(self.attrs.get("technology"),
                                self.attrs.get("region"),
                                self.attrs.get("carrier"),
                                self.attrs.get("scenario"),
                                self.attrs.get("time"),
                                self.attrs.get("subtechnology"),
                                self.attrs.get("subscenario"))

