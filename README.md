# FER project

## Objective

The goal of this project is to perform **Facial Emotion Recognition (FER)** using a **combination of multiple datasets**, including both *in-the-wild* and *studio-controlled* images.  
By mixing **RAF-DB**, **AffectNet (subset)**, and **KDEF**, the objective is to build a robust model capable of generalizing across diverse facial expressions and acquisition conditions, while achieving the best possible performance.

---

## Datasets Used

This project relies on the following publicly available datasets:

- **KDEF (Karolinska Directed Emotional Faces)**  
  Studio-controlled facial expression dataset  
  https://www.kaggle.com/datasets/chenrich/kdef-database?select=neutral

- **RAF-DB (Real-world Affective Faces Database)**  
  Large-scale dataset collected in real-world conditions  
  https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/

- **AffectNet (Subset)**  
  In-the-wild facial expressions with large variability in pose, illumination, and occlusion  
  https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format


## Execution

This project uses **YAML configuration files** to define all training and evaluation parameters, including model architecture, datasets, hyperparameters, and paths.  
Example configuration files are provided in the `configs/` directory.

### 1. Create a Configuration File

Before launching training, create a new configuration file by:

- Copying one of the existing examples in `configs/`
- Modifying it according to your needs (model, dataset, batch size, learning rate, etc.)

### 2. Launch Training

From the **root of the project**, start training using the desired configuration file:

```bash
python -m src.training.train --config "./configs/<your_config_file>"
```

### 3. Run Evaluation
Once training is complete, evaluate the trained model using the same configuration file:

```bash
python -m src.training.evaluate --config "./configs/<your_config_file>"
```