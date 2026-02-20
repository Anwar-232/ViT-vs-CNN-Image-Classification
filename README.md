# ViT-vs-CNN-Image-Classification
# Vision Transformer (ViT) vs. CNN: Image Classification Benchmark

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Computer%20Vision-blue?style=for-the-badge" alt="CV">
  <img src="https://img.shields.io/badge/MNIST-green?style=for-the-badge" alt="MNIST">
</p>

##  Overview
This project explores the application of **Transformer architectures** in Computer Vision. It provides a direct comparison between a fine-tuned **Vision Transformer (ViT)** and a traditional **Convolutional Neural Network (CNN)** baseline using the MNIST dataset.

The goal is to analyze the trade-off between **Classification Accuracy** and **Computational Efficiency (Training Time)** when using large pre-trained models versus lightweight custom architectures.



---

##  Methodology & Models

### 1. Vision Transformer (ViT)
- **Model:** `google/vit-base-patch16-224-in21k` (Pre-trained on ImageNet-21k).
- **Strategy:** Transfer Learning / Fine-Tuning for 3 epochs.
- **Preprocessing:** Images upscaled to $224 \times 224$ and converted to 3 RGB channels.

### 2. Simple CNN
- **Architecture:** Custom-built with 1 Convolutional layer, MaxPooling, and a Fully Connected head.
- **Strategy:** Training from scratch for 3 epochs.
- **Preprocessing:** Kept at original $28 \times 28$ Grayscale format.

---

## Results Comparison
The evaluation was conducted on a fixed subset of 4,000 training samples and 1,000 test samples to ensure a fair and efficient benchmark.

| Metric | Vision Transformer (ViT) | Simple CNN | Observation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **98.10%** | 91.50% | ViT gained **+6.6%** accuracy. |
| **Training Time** | 441.03 sec | **3.76 sec** | CNN was **~117x faster**. |

###  Discussion
* **Accuracy Dominance:** ViT's superior accuracy is attributed to **Transfer Learning**, leveraging high-level features learned from millions of images during pre-training.
* **Efficiency Trade-off:** While CNN is less accurate, its extreme speed and low resource consumption make it ideal for edge devices or real-time applications where training time is a bottleneck.

---

##  Key Conclusions
- **ViT** is the preferred choice for tasks where maximum precision is critical, and hardware acceleration (GPU) is available.
- **CNNs** remain the optimal baseline for simple image tasks and resource-constrained environments.

##  How to Run
1.  Open `ViT_vs_CNN_Classification.ipynb` in Google Colab.
2.  Ensure you have a GPU runtime selected.
3.  Install dependencies: `pip install transformers torchvision scikit-learn matplotlib`.
4.  Run the notebook to reproduce the comparative plots and metrics.
