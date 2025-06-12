# 🏷️ What's ADE? (Auto Data Embedding) - in Progress
<br/>

*"Train. Detect. Automate."* <br/><br/>

**A PyQt-based desktop app for automatic image labeling and embedding extraction using detection and vision models.**

This tool helps you annotate image datasets quickly using pretrained models (e.g., YOLOv8~), modify labels with a GUI, and extract embeddings for downstream tasks.

<br/><br/><br/>

Main View <br/>
![Screenshot](./assets/Demo_250612(1).png)  <br/><br/>
Auto Embedding. . . <br/>
![Screenshot](./assets/Demo_250612(2).png)  <br/><br/>
Train hyperparameter setting <br/>
![Screenshot](./assets/Demo_250612(3).png)

<br/><br/><br/>

# 📦 Features (planned)
✅ Automatic object detection & YOLO format label generation

✅ Visual label editing using PyQt GUI

✅ Embedding extraction (e.g., with CLIP or ResNet)

✅ Save embeddings as .npy, .json

✅ Support for .jpg, .png, .txt YOLO datasets

<br/><br/><br/>
# 🔧 Setup Instructions

### 1. Download Release version

| Release Date | Version |
|------|------|
| 2025-06-12 | [v1.0.2](https://github.com/ohsopp/YoloADE/releases/tag/v1.0.2) |
| 2025-06-10 | [v1.0.1](https://github.com/ohsopp/YoloADE/releases/tag/v1.0.1) |
| 2025-06-05 | [v1.0.0](https://github.com/ohsopp/YoloADE/releases/tag/v1.0.0) |


### 2. Create and Activate a Virtual Environment
macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```

Windows
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages
```
pip install -r requirements.txt
```

Example `requirements.txt`
```
PyQt5==5.15.11
opencv-python==4.11.0.86
ultralytics==8.3.145
numpy==1.24.4
pillow==10.4.0
matplotlib==3.7.5
```

### 4. Install CUDA pytorch
```
pip install torch==2.4.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

<br/>

<br/><br/><br/>
# ▶️ How to Build
```
pyinstaller main.py --onedir --noconsole --clean --icon=icons/icon.ico --add-binary "yolo.exe;."
```

<br/><br/><br/>
# 📁 Project Structure (draft)
```
pyqt_app/
├── build_guide.txt               # Build guide
├── main.py                       # App entry point
├── data_integrator.py
├── training_settings_dialog.py
├── icons/                        # PyQt GUI icons
├── model/                        # YOLO, CLIP, etc. wrappers
├── dist/
│   └── main.exe                  # .exe file
├── dataset/
│   ├── images/
│   └── labels/
├── requirements.txt              # Requirements package settings
└── README.md
```

<br/><br/><br/>
# 🧠 Model Support
- [x] YOLOv8 via Ultralytics  
- [ ] CLIP (planned)  
- [ ] ResNet (planned)  
- [ ] ONNX custom model support (planned)

<br/><br/><br/>
# 🙌 Contributing
Issues, feedback, and pull requests are welcome!
Open a GitHub issue or fork this repo to contribute.

<br/><br/>


