from pathlib import Path
from omegaconf import OmegaConf
from match_free_pose import FreePose
from match_aligned_pose import AlignedPose

if __name__ == "__main__":
    # Load configuration
    cfg_path = Path("/home/huangyue/Mycodes/MetaScenes/scripts/pose_alignment/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # Select alignment tool based on pose type
    pose_type = "free"
    if pose_type == "free":
        alignment_tool = FreePose(cfg['freepose'])
    elif pose_type == "aligned":
        alignment_tool = AlignedPose(cfg['alignedpose'])
    else:
        raise ValueError("Invalid pose_type")

    # Define scan list
    scan_list = ['scene0001_00']

    # Run alignment tool for each scan
    for one_scan in scan_list:
        print(one_scan)
        alignment_tool.run(one_scan, save_output_mesh=False, show_output_mesh=True)