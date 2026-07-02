# AI-Powered Virtual Try-On (IDM-VTON)

This project explores the implementation of the **IDM-VTON** model to perform virtual try-on, allowing a garment image to be realistically "worn" by a person in a photograph. This repository documents the end-to-end development journey of deploying a highly resource-intensive AI model, including the infrastructure challenges faced and the architectural solutions implemented.

## The Technical Journey

### 1. Local Development
* **Goal:** Run the IDM-VTON model on a local workstation.
* **Outcome:** The model requires a dedicated NVIDIA GPU with high VRAM (12GB+). My local machine, using integrated graphics, lacked the necessary hardware resources to initialize the model.

### 2. Cloud-Based Prototyping (Google Colab)
* **Goal:** Move the project to a cloud GPU environment using the `IDM_VTON_Colab (5)_2.ipynb` notebook.
* **Outcome:** Using the T4 GPU in Google Colab, I successfully ran the model after applying memory optimizations, such as resolution scaling (down to 576x768) and enabling VAE tiling/slicing to prevent memory crashes[cite: 2].

### 3. API & Proxy Development
To make this model accessible to other applications, I developed a dual-part API system:
* **`fastapi_app.py`**: A backend service that runs on Google Colab. It loads the AI model into GPU memory once and provides an endpoint (`/tryon`) to process images.
* **`local_proxy_app.py`**: A local script that acts as a secure "bridge." Because Colab sessions are temporary, this local proxy allows you to keep a stable local interface that forwards requests to the cloud-hosted API.

### 4. Memory & Performance Limitations
During deployment, I observed that the IDM-VTON model is extremely memory-intensive:
* The API frequently experiences session crashes due to memory limits (CUDA Out of Memory) on standard T4 GPU instances. 
* **Conclusion:** This architecture is functionally correct, but for stable, production-grade performance, it requires high-end hardware with at least 24GB of dedicated VRAM (e.g., A100 or L4 GPUs). Users with powerful local GPUs can run this entire stack locally by setting `DEVICE = "cuda"` in the configuration[cite: 4].

## Project Directory Structure

```text
idm-vton-project/
├── IDM_VTON_Colab.ipynb # The notebook used for Colab-based testing
├── fastapi_app.py  # The main API that processes image requests
├── local_proxy_app.py  # The bridge between your local machine and Colab
├── OutputImage.png    # Sample output demonstrating the model's success
└── README.md           # Project documentation