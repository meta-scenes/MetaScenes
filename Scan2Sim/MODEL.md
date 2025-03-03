# **Training and Inference for Scan2Sim**  

## **What is Scan2Sim?**  
**Scan2Sim** is a multi-modal alignment model designed to retrieve the most optimal asset candidate from a set of candidates, leveraging ground truth optimal asset selection annotations from **METASCENES**.  

---

## **Environment Setup**  
To set up the required environment for **SceneVerse**, follow these steps:  

```bash
$ conda env create -n metascenes python=3.10
$ conda activate metascenes
$ pip install -r requirements.txt
```  

---

## **Pre-trained Model and Training Data Download**  
Download the datasets and pre-trained models from [here](..).  
Once downloaded, update the relevant file paths accordingly:  

```bash
# Modify the following paths in ./data/Ti_anno.yaml
DATA_PATH: "/path/to/training/data"
PC_PATH: "/path/to/pointcloud"
IMAGE_PATH: "/path/to/images"
```  

---

## **Running Experiments**  

### **1. Training**  
To train **Scan2Sim**, use the following command:  

```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model Scan2Sim_PointBERT_Colored --npoints 1024 --lr 3e-3 --output-dir ./outputs/mytrain
```  
This command launches a distributed training job across 8 GPUs, using **Scan2Sim_PointBERT_Colored** as the model and setting the learning rate to `3e-3`.

---

### **2. Inference**  
To test a trained checkpoint, use the following command:  

```bash
# Single-GPU inference
$ python main.py --model Scan2Sim_PointBERT_Colored --batch-size 4 --lr 3e-3 --npoints 1024 \
  --output-dir release_test --pretrain_dataset_name ti_anno --validate_dataset_name ti_anno \
  --evaluate_3d --resume /path/to/best/checkpoint
```  
This runs inference on a single GPU with a batch size of `4`, evaluating the model on **3D data** using the best checkpoint.

---

## **Notes**  
- Ensure that all paths in `Ti_anno.yaml` are updated before running training or inference.  
- Adjust the batch size and number of GPUs as needed, depending on your system's hardware.  
- If resuming from a checkpoint, make sure the `--resume` path points to a valid pre-trained model.  

This guide provides everything needed to train and evaluate **Scan2Sim** efficiently. 🚀