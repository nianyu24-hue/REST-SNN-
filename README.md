<div align="center">

# REST-SNN: Robust and Efficient Spatio-Temporal Attention for Training Spiking Neural Networks

<img width="979" height="518" alt="Framework Overview" src="https://github.com/user-attachments/assets/8a81b068-90e3-4e25-a7df-218b0aad6fbd" />

</div>

---

This repository provides the official implementation of **REST-SNN**, a robust and efficient framework for **Spiking Neural Networks (SNNs)**. REST-SNN integrates **AI-LIF neurons** with a **Dual Regularization strategy (SSR & TCR)** to achieve state-of-the-art robustness against environmental perturbations.

The codebase is released to support full reproducibility of the experimental results reported in the paper.

---

## Requirements

The code has been tested with the following environment:

- Python >= 3.8  
- PyTorch >= 1.10.0  
- CUDA >= 11.3  
- SpikingJelly *(optional, if used as backend)*  

---

## Installation

```bash
git clone https://github.com/your-username/REST-SNN.git
cd REST-SNN
pip install -r requirements.txt
```
---

##  Various perturbations defined in observation_noise.py

- Including Rotation,Shift,Color Jitter,Cutout and Erasing
---

##  Training


The training configuration (including $\lambda_1$ for SSR and $\lambda_2$ for TCR) is centralized in run.sh.

```
bash run.sh
```

---
## Evaluation

```
python test.py 
```
---
