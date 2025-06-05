# 🏷️ ADE(Auto Data Embedding)
**A PyQt-based desktop app for automatic image labeling and embedding extraction using detection and vision models.**

This tool helps you annotate image datasets quickly using pretrained models (e.g., YOLOv8~), modify labels with a GUI, and extract embeddings for downstream tasks.

<br/><br/><br/>

# 📦 Features (planned)
✅ Automatic object detection & YOLO format label generation

✅ Visual label editing using PyQt GUI

✅ Embedding extraction (e.g., with CLIP or ResNet)

✅ Save embeddings as .npy, .json

✅ Support for .jpg, .png, .txt YOLO datasets

<br/><br/><br/>
# 🔧 Setup Instructions

## 1. Clone the repository
```
git clone https://github.com/ohsopp/AutoLabelEmbed.git
cd AutoLabelEmbed
```

## 2. Create and Activate a Virtual Environment
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

## 3. Install Required Packages
```
pip install -r requirements.txt
```

<br/>
Example `requirements.txt`
```
pyqt5
opencv-python
ultralytics
torch
numpy
Pillow
matplotlib
```

<br/><br/><br/>
# ▶️ How to Run
```
python main.py
```

<br/><br/><br/>
# 📁 Project Structure (draft)
```
AutoLabelEmbed/
├── main.py                  # App entry point
├── ui/                      # PyQt GUI logic
├── model/                   # YOLO, CLIP, etc. wrappers
├── utils/                   # Helpers (I/O, label formatting)
├── dataset/
│   ├── images/
│   └── labels/
├── embeddings/              # Saved embeddings
├── requirements.txt
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

