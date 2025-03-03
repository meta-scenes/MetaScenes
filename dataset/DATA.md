
## **What is MetaScenes?**  

**MetaScenes** provides high-quality 3D scenes annotated by humans and optimized by physics-based simulations. 

---

## **1. Data Download**  

We currently host our dataset on Google Drive. To request access, please fill out [this form](https://forms.gle/k94RWbdoQ1KrLZpF7).  

### **Dataset Structure**  

After downloading and extracting the dataset, you will see the following folder, which includes:  

1. **Human-Annotated Scenes** – 3D scenes with object placements manually annotated.  
2. **Precomputed Transformation Matrices** – Optimized transformations for creating physically plausible scenes. You can use this [script](https://github.com/yuhuangyue/MetaScenes/blob/main/dataset/export_and_visualize_phy_scene.py) to convert an annotated scene to a physics-based scene.
3. **Preprocessed Scene Layouts** – Scene layouts generated using both heuristic-based and model-based approaches for visualization.  

We recommend using **Human-Annotated Scenes**, as physics-optimized scenes may sometimes be unstable, leading to incorrect object placements.

The folder structure is as follows:

```shell
MetaScenes/
|-- Annotation_scenes                 # 1. Human-annotated 3D scenes
  |-- scan_id/
    |-- inst_id/        
      |-- mesh.obj                     # 3D object mesh
      |-- material.mtl                  # Material information
      |-- texture.png                    # Texture file
|-- Physical_simu                      # 2. Precomputed Transformation Matrices
  |-- scan_id.json                      # JSON file with transformation matrices
|-- Layout                              # 3. Precomputed scene layouts
  |-- Heuristic/                        # Layouts generated using heuristic methods
    |-- scan_id.json
  |-- RoomFormer/                        # Layouts generated using RoomFormer model
    |-- scan_id.json
```


---

## **2. Data Visualization**  

To visualize and interact with the dataset, follow the steps below.  

### **2.1. Installation**  

We tested our visualization scripts on **Ubuntu 22.04** with **NVIDIA CUDA 12.3**. To set up the environment, run:  

```shell
conda create -n sceneverse python=3.9
conda activate sceneverse
pip install trimesh numpy argparse
```

---

### **2.2. Visualizing Annotation Scenes**  

We provide a script to visualize annotation scenes with optional bounding boxes and small objects. The script supports loading layouts using either **heuristic-based** or **model-based** methods.  

#### **Usage**  
```shell
python dataset/visualize_anno_scene.py --scan scene0001_00 \
                          --save_path /path/to/Annotation_scenes \
                          --layout_path_heuristic /path/to/heuristic_layout \
                          --layout_path_model /path/to/model_layout \
                          --layout_type model \
                          --show_small_objects \
                          --show_bbox
```
💡 **Note:** Ensure that the specified directories contain the necessary scene data and layout files.  

---

### **2.3. Visualizing Physically Optimized Scenes**  

We also provide a script to visualize physically optimized scenes.  

#### **Usage**  
```shell
python dataset/export_and_visualize_phy_scene.py --one_scan scene0001_00 \
                                      --simu_matrix_info_path /path/to/Physical_simu/scene0001_00.json \
                                      --dataset_path /path/to/Annotation_scenes/
```
This will display the transformed 3D scene with objects placed according to the precomputed physical transformations.
You can also export these objects mesh by adding ```--export_dir /path/to/export/.obj/file```

---

## **3. Blender Visualization**  

For higher-quality rendering and interacting, we provide a Blender script to visualize physically optimized scenes.  

### **3.1. Install Blender**  

Before using the Blender script, ensure Blender is installed. You can download it from the official website:  

🔗 [Download Blender](https://www.blender.org/download/)  

### **3.2. Running the Script**  

1. Open Blender.  
2. Copy the script from [`dataset/visualize_phy_scene_blender.py`](...) into the Blender scripting editor.  
3. Run the script to load the scene with optimized object placements.  

This method provides **better rendering quality** and allows for further scene modifications inside Blender.  

---
