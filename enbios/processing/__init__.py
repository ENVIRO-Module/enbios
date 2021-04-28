from typing import Dict, Tuple, List

from nexinfosys.command_generators import Issue
from nexinfosys.embedded_nis import NIS


def read_parse_configuration(file_name) -> Dict:
    """
    (read &) Parse the configuration file. See "example_config.yml"

    :param file_name: Name of the configuration file
    :return: If mandatory keys are present it returns a "dict" with every variable in the configuration file
    """
    def read_yaml_configuration(file_content) -> Dict:
        import yaml
        d = yaml.load(file_content, Loader=yaml.FullLoader)
        return d

    if isinstance(file_name, str):
        # Read into string
        with open(file_name, 'r') as file:
            contents = file.read()

        # Read depending on format
        if file_name.endswith("yaml") or file_name.endswith("yml"):
            cfg = read_yaml_configuration(contents)
    elif isinstance(file_name, dict):
        cfg = file_name

    # Check configuration
    mandatory_keys = ["nis_file_location",
                      "correspondence_files_path",
                      "simulation_type",
                      "simulation_files_path",
                      "lci_data_locations",
                      "output_directory"]
    any_not_found = False
    for k in mandatory_keys:
        if k not in cfg:
            print(f"{k} key not found in the configuration file {file_name}")
            any_not_found = True
    if any_not_found:
        raise Exception("Configuration file incomplete or incorrect. See log for missing keys")
    return cfg


def read_prepare_nis_file(nis_file_url: str) -> Tuple[NIS, List[Issue]]:
    nis = NIS()
    nis.open_session(True)
    nis.reset_commands()
    # Load file and execute it
    nis.load_workbook(nis_file_url)
    issues = nis.submit_and_solve()
    # tmp = nis.query_available_datasets()
    # print(tmp)
    # d = nis.get_results([(tmp[0]["type"], tmp[0]["name"])])
    return nis, issues