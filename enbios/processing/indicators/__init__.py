import math

from nexinfosys.model_services import State

materials = {
    "Aluminium",
    "Antimony",
    "Arsenic",
    "Baryte",
    "Beryllium",
    "Borates",
    "Cadmium",
    "Cerium",
    "Chromium",
    "Cobalt",
    "Copper",
    "Diatomite",
    "Dysprosium",
    "Europium",
    "Fluorspar",
    "Gadolinium",
    "Gallium",
    "Gold",
    "Gypsum",
    "IronOre",
    "KaolinClay",
    "Lanthanum",
    "Lead",
    "Lithium",
    "Magnesite",
    "Magnesium",
    "Manganese",
    "Molybdenum",
    "NaturalGraphite",
    "Neodymium",
    "Nickel",
    "Palladium",
    "Perlite",
    "Phosphorus",
    "Platinum",
    "Praseodymium",
    "Rhenium",
    "Rhodium",
    "Samarium",
    "Selenium",
    "SiliconMetal",
    "Silver",
    "Strontium",
    "Sulphur",
    "Talc",
    "Tantalum",
    "Tellurium",
    "Terbium",
    "Tin",
    "Titanium",
    "Tungsten",
    "Vanadium",
    "Yttrium",
    "Zinc",
    "Zirconium"
}


def supply_risk(state: State):
    sr = 0
    for i in materials:
        ri = state.get(i)
        if ri is not None:
            SRi = state.get(f"sr{i}")
            ci = state.get(f"c{i}")
            sr += ri*SRi/ci
    return sr


def recycling_rate(state: State):
    rr_num = 0
    rr_denom = 0
    for i in materials:
        ri = state.get(i)
        if ri is not None:
            RRi = state.get(f"rr{i}")
            rr_num += ri*RRi
            rr_denom += ri
    if rr_denom != 0.0:
        return rr_num / rr_denom
    else:
        return -1.0
