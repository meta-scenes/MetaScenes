# **Preprocessing for MetaScenes**

MetaScenes provides a comprehensive pipeline to construct replica scenes of real-world environments.

Here, we provide the code of the following three key preprocessing components:

1. **Heuristic-based Room Layout Estimation**
2. **Object Pose Alignment**
3. **Physics-based Scene Optimization**

---

## **Environment Setup**

To run the MetaScenes Processing, you will need to set up the required Python environment and install dependencies.

### **Install Environment:**

1. **Create Conda Environment:**

   ```bash
   $ conda env create -n metascenes python=3.9
   $ conda activate metascenes
   ```

2. **Install Required Packages:**

   Install the necessary Python dependencies from the provided `requirements.txt`:

   ```bash
   $ pip install -r requirements.txt
   ```

### **Blender Setup:**

MetaScenes relies on **Blender** for physics-based optimization. Make sure you have **Blender** installed on your system. Download it from the [official Blender website](https://www.blender.org/download/).

- **Ensure `bpy` is installed** and compatible with your Python environment. If you do not need Blender-specific features, you can use the `fake-bpy-module`:

   ```bash
   pip install fake-bpy-module
   ```

- If you need Blender's specific features, install the `bpy` module for your version of Blender:

   ```bash
   pip install bpy==<blender_version>
   ```

---

## **1. Heuristic-based Room Layout Estimation**

This script processes 3D scan data to generate floor layouts from annotated scenes. You can visualize the results and save them as JSON files.

### **Basic Usage:**

To process a scan, save the result to a JSON file, and visualize the scene, run the following command:

```bash
python preprocess/layout_estimation/heuristic_layout.py --scans scene0001_00 \
                         --dataset_path /path/to/annotated/scenes \
                         --output_dir /path/to/output \
                         --visualize \
                         --padding 0.05
```


### **Output:**


The script generates JSON files for each scan in the specified output directory. These files contain floor layout data, with each JSON file named `<scan_id>.json`:

```bash
/path/to/output/scene0001_00.json
/path/to/output/scene0002_00.json
```

Each file includes information about the vertices of the floor layout. We also provide precomputed scene layouts. You can download them by referring to [DATA.md](https://github.com/yuhuangyue/MetaScenes/blob/main/dataset/DATA.md).

---

## **2. Object Pose Alignment**

This script allows you to align meshes to the 3D scene by adjusting their pose. It supports two types of pose alignments:

- **Free Pose Alignment**: Aligns the mesh based on the camera pose used during scene reconstruction.
- **Fixed Pose Alignment**: Aligns the mesh based on an axis-aligned orientation (e.g., pre-processed).

### **Free Pose Alignment:**

In this mode, the mesh is aligned based on the camera pose from the **single image** used during the scene reconstruction.


#### **Example Command:**

```bash
python preprocess/pose_alignment/run.py --config /path/to/config.yaml \
                              --pose_type free \
                              --scans scene0001_00 scene0002_00 \
                              --save_output --show_output
```

### **Fixed Pose Alignment:**

In this mode, the mesh's initial pose is axis-aligned. This is typically used when the mesh has already been manually positioned.


#### **Example Command:**

```bash
python preprocess/pose_alignment/run.py --config /path/to/config.yaml \
                              --pose_type fixed \
                              --scans scene0001_00 scene0002_00 \
                              --save_output
```

### **Output:**

For each scan, the script outputs an aligned mesh. If `--save_output` is provided, the aligned mesh is saved to disk. If `--show_output` is specified, the aligned mesh is displayed interactively.
We provide some test cases in ```process/pose_alignment/examples```.
---

## **3. Physics-based Scene Optimization**

The physics-based scene optimization process consists of three main steps:

1. **Scene Graph Generation**: Creates a scene graph from the scene data.
2. **Local Optimization**: Optimizes objects in the scene using the scene graph.
3. **Global Optimization**: Optimizes the entire scene, ensuring physical plausibility.

### **Run the Pipeline:**

To run the full physical optimization pipeline, use the following command:

```bash
python preprocess/physical_opimization/metascenes_phy.py --config /path/to/config.yaml --blender_exec /usr/local/bin/blender
```


### **Output:**

After completing the pipeline, the results are saved as:

- Scene Graphs: Saved in the `ssg_save_dir`.
- Local Optimization Results: Saved in the `local_opt_save_dir`.
- Global Optimization Results: Saved in the `global_opt_save_dir`. We also provide precomputed global optimization Results. You can download them by referring to [DATA.md](https://github.com/yuhuangyue/MetaScenes/blob/main/dataset/DATA.md).

Example output files:

```bash
/path/to/save/scene_graphs/scene0001_00.json
/path/to/save/local_optimization_results/scene0001_00.json
/path/to/save/global_optimization_results/scene0001_00.json
```
