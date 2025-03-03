import argparse
from pathlib import Path
from omegaconf import OmegaConf
from match_free_pose import FreePose
from match_fixed_pose import FixedPose

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run pose alignment tool.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--pose_type", type=str, choices=["free", "fixed"], required=True, help="Type of pose alignment: 'free' or 'fixed'.")
    parser.add_argument("--scans", nargs='+', required=True, help="List of scan IDs to process.")
    parser.add_argument("--save_output", action='store_true', help="Flag to save the output mesh.")
    parser.add_argument("--show_output", action='store_true', help="Flag to display the output mesh.")
    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    cfg = OmegaConf.load(cfg_path)

    # Select alignment tool based on pose type
    if args.pose_type == "free":
        alignment_tool = FreePose(cfg['freepose'])
    else:
        alignment_tool = FixedPose(cfg['fixedpose'])

    # Run alignment tool for each scan
    for one_scan in args.scans:
        print(f"Processing scan: {one_scan}")
        alignment_tool.run(one_scan, save_output_mesh=args.save_output, show_output_mesh=args.show_output)

if __name__ == "__main__":
    main()