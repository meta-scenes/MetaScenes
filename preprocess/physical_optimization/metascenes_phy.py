
import yaml
from ssg_generation import main as ssg_generation_run
import subprocess
import os
import argparse

class MetaScenes_Phy():
    """
    End-to-end pipeline for running physical optimization of MetaScenes.
    """

    def __init__(self, config_path):
        """
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.config = config

    def run(
            self,
            blender_exec="blender",

    ):
        """
        Executes physical optimization of MetaScenes, running the following steps:
        1. Scene graph generation
        2. Local optimization based on scene graph
        3. Global optimization for the whole scene
        """

        # Load parameters from config
        scan_list = self.config["scan_list"]
        dataset_dir = self.config["dataset_path"]
        ssg_save_dir = self.config["ssg_save_dir"]
        local_opt_save_dir = self.config["local_opt_save_dir"]
        global_opt_save_dir = self.config["global_opt_save_dir"]
        inst_name_dir = self.config["inst_name_dir"]


        print(f"""

        {"#" * 50}
        {"#" * 50}
        # Running Step 1 -- Scene graph generation
        {"#" * 50}
        {"#" * 50}

                                """)

        save_path_list = ssg_generation_run(scan_list, ssg_save_dir, inst_name_dir, dataset_dir)

        for ssg_one_scan_path in save_path_list:
            if os.path.exists(ssg_one_scan_path) and os.path.getsize(ssg_one_scan_path) > 0:
                print(f"File saved successfully at {ssg_one_scan_path}")
            else:
                print("File save failed!")
                raise ValueError("Failed Step 1!")

        print(f"""

        {"#" * 50}
        {"#" * 50}
        # Running Step 2 -- Local optimization based on scene graph
        {"#" * 50}
        {"#" * 50}

                                """)

        command = [blender_exec, "--background", "--python", 'process/physical_optimization/ssg_optim_simu.py']

        process = subprocess.run(command, capture_output=True, text=True)

        # if process.stderr:
        #     print("Error Output:")
        #     print(process.stderr)
        #     raise ValueError("Failed Step 2!")
        for one_scan in scan_list:
            if not os.path.exists(os.path.join(local_opt_save_dir, f'{one_scan}_invalid_objs.json')) \
                    or not os.path.exists(os.path.join(local_opt_save_dir, f'{one_scan}_simu_matrix_info_support.json')):
                raise ValueError("Failed Step 2!")
            else:
                print(f"{scan_list} | File saved successfully at {local_opt_save_dir}")




        print(f"""

        {"#" * 50}
        {"#" * 50}
        # Running Step 3 -- Global optimization for whole scene
        {"#" * 50}
        {"#" * 50}

                                """)

        command = [blender_exec, "--background", "--python", 'process/physical_optimization/run_simu.py']

        process = subprocess.run(command, capture_output=True, text=True)


        # if process.stderr:
        #     print("Error Output:")
        #     print(process.stderr)
        #     raise ValueError("Failed Step 3!")

        for one_scan in scan_list:
            if not os.path.exists(os.path.join(global_opt_save_dir, f'{one_scan}.json')):
                raise ValueError("Failed Step 3!")
            else:
                print(f"File saved successfully at {os.path.join(global_opt_save_dir, f'{one_scan}.json')}")


        print("All steps completed successfully!")



def main():
    """
    Main function to parse command-line arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="Run MetaScenes Physical Optimization Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument("--blender_exec", type=str, required=True, help="Path to the Blender executable")


    args = parser.parse_args()

    # Create pipeline and run
    pipeline = MetaScenes_Phy(config_path=args.config)
    pipeline.run(blender_exec=args.blender_exec)


if __name__ == "__main__":
    main()

