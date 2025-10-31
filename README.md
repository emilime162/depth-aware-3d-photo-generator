# 🎥 Depth-Aware 3D Photo Generator

**Author:** Emily Chen  
**Framework:** Gradio + PyTorch + Hugging Face Transformers  
**Model:** [Intel/dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas)

---

## 🧠 Overview
This app transforms a single 2D photo into a **cinematic 3D parallax animation** using depth estimation and motion synthesis.  
It predicts depth maps with **MiDaS** and applies **foreground–background separation, dynamic zoom, and custom bokeh effects** for realistic motion.

---

## 🚀 Live Demo
Try it directly on **Hugging Face Spaces**:  
👉 [https://huggingface.co/spaces/emilime28/3d-photo-generator](https://huggingface.co/spaces/emilime28/3d-photo-generator)

---

## ⚙️ Requirements
See [`requirements.txt`](./requirements.txt) for dependencies.  
If you’d like to run locally:

```bash
git clone https://github.com/emilime162/depth-aware-3d-photo-generator.git
cd depth-aware-3d-photo-generator
pip install -r requirements.txt
python app.py
