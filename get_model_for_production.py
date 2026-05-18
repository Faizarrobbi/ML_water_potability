import os
import yaml
import shutil
import argparse


def get_model_for_prod(ARGS):
    dir_reg_models = os.path.join(
        ARGS.dir_mlflow_logs,
        "models",
        ARGS.model_name
    )

    print(dir_reg_models)

    list_versions = [
        version
        for version in os.listdir(dir_reg_models)
        if os.path.isdir(os.path.join(dir_reg_models, version))
    ]

    print(list_versions)

    version_base = list_versions[0].split("-")[0]
    num_versions = len(list_versions)

    latest_version = f"{version_base}-{num_versions}"

    file_yaml = os.path.join(
        dir_reg_models,
        latest_version,
        "meta.yaml"
    )

    with open(file_yaml, "r") as file_hd:
        dict_yaml = yaml.safe_load(file_hd)

    target_model_for_prod = (
        dict_yaml["storage_location"]
        .replace("file:///", "")
        .replace("/", "\\")
    )

    print(target_model_for_prod)

    list_files_for_prod = os.listdir(target_model_for_prod)

    os.makedirs(ARGS.dir_model_for_prod, exist_ok=True)

    for _file in list_files_for_prod:
        src_path = os.path.join(target_model_for_prod, _file)
        dst_path = os.path.join(ARGS.dir_model_for_prod, _file)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

    print(
        f"Copied files to directory: {ARGS.dir_model_for_prod}"
    )


def main():
    model_name = "water_potability"
    dir_mlflow_logs = "mlruns"
    dir_model_for_prod = "model_for_production"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_name",
        default=model_name,
        type=str,
    )

    parser.add_argument(
        "--dir_mlflow_logs",
        default=dir_mlflow_logs,
        type=str,
    )

    parser.add_argument(
        "--dir_model_for_prod",
        default=dir_model_for_prod,
        type=str,
    )

    ARGS, unparsed = parser.parse_known_args()

    get_model_for_prod(ARGS)


if __name__ == "__main__":
    main()