from enbios.common.helper import generate_workbook
from enbios.input.simulators.sentinel import SentinelSimulation


def sentinel_to_prep_file(sentinel_package_index, output):
    sim = SentinelSimulation(sentinel_package_index)
    cmds = sim.generate_template_data()
    s = generate_workbook(cmds, True)
    with open(output, "wb") as f:
        f.write(s)
