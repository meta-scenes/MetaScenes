<h2 align="center">
  <b>METASCENES: Towards Automated Replica Creation for Real-world 3D Scans</b>

  <b><i>CVPR 2025 </i></b>
</h2>

<p align="center">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Video-green?style=plastic&logo=arXiv&logoColor=green' alt='Video'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

<p align="center">
    <a href="">Huangyue Yu</a>,
    <a href="">Baoxiong Jia</a>,
    <a href="https://yixchen.github.io/">Yixin Chen</a>,
    <a href="">Yandan Yang</a>,
    <a href="https://xiaoyao-li.github.io/">Puhao Li</a>,
    <a href="">Rongpeng Su</a>,
    <br>
    <a href="">Jiaxin Li</a>,
    <a href="">Qing Li</a>,
    <a href="">Wei Liang</a>,
    <a href="https://zhusongchun.net/">Song-Chun Zhu</a>,
    <a href="">Tengyu Liu</a>,
    <a href="https://siyuanhuang.com/">Siyuan Huang</a>
</p>

<p align="center">
    <img src="assets/teaser4-1.png" width=90%>
</p>

PhyRecon harnesses both differentiable rendering and differentiable physics simulation to achieve physically plausible scene reconstruction from multi-view images.

## News

- [2025/1/14] Code is released. For more information, please visit our [project page](https://phyrecon.github.io/)!

## Installation
```bash
conda create -n metascenes python=3.9
conda activate metascenes
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data
Please download the preprocessed [data](https://drive.google.com/drive/folders/1gdYiCg9VFlFD5dLuJQz2j90UB80oohc2?usp=drive_link) and unzip in the `data` folder. The resulting folder structure should be:
```
└── PhyRecon
  └── data
    ├── replica
        ├── scan ...
    ├── scannet
        ├── scan ...
    ├── scannetpp
        ├── scan ...
```
Since the full dataset is quite large, we use the `scannetpp/scan1` as an example scene. Please download this scene [here](https://drive.google.com/file/d/1rUXLZO5c2LFHapN7TYQPaikrihHYzeHC/view?usp=drive_link) and ensure it is placed under `data/scannetpp` following the folder structure.

## Training


## Dataset Construction
### 1. Object pose alignment
```bash
# This script runs a pose alignment tool using either "free" or "aligned" pose estimation.
python pose_alignment.py --config /path/to/config.yaml --pose_type free --scans scene0001_00 scene0002_00 --save_output --show_output
```

### 2. Room layout estimation
This script processes scene data to generate floor boundary information and optionally visualize it.
```bash
# To generate floors for scene0001_00 with default padding and visualization enabled:
python heuristic_layout.py --scans scene0001_00 \
    --anno_path /mnt/fillipo/huangyue/recon_sim/7_anno_v2/anno_info_ranking_v2.json \
    --output_dir /home/huangyue/Mycodes/MetaScenes/scripts/layout_estimation/heuristic_layout \
    --visualize


```
### 3. Physics-based optimization



## Acknowledgements
Some codes are borrowed from ULIP2. We thank all the authors for their great work. 

## Citation

```bibtex

```