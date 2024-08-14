import os
import json

def read_config(config_file: str) -> dict:

    with open(config_file) as file:
        config_data = json.load(file)

    for field in ["data_dir", "data_root", "output_dir", "input_xrange", "input_yrange"]:
        if field not in config_data:
            raise Exception("Missing field {field} in {config_file}.")

    if not os.path.isdir(config_data["output_dir"]):
        os.mkdir(config_data["output_dir"])

    return config_data
