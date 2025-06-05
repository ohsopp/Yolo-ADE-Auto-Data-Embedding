import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO ADE - Auto Data Embedding")
        self.resize(800, 600)

        # 모델 로드
        self.model = YOLO("runs/detect/train5_32/weights/best.pt")

        # UI 구성
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: black")

        self.open_btn = QPushButton("영상 열기")
        self.pause_btn = QPushButton("⏸ 중지")
        self.pause_btn.setEnabled(False)

        self.open_btn.clicked.connect(self.open_video)
        self.pause_btn.clicked.connect(self.toggle_pause)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.pause_btn)

        layout = QVBoxLayout()
        layout.addLayout(btn_layout)

        # 영상 라벨 가운데 정렬을 위한 레이아웃 추가
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        layout.addLayout(video_layout)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_paused = False

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "MP4 files (*.mp4);;All files (*)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.is_paused = False
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("⏸ 중지")
            self.timer.start(30)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText("▶ 재생")
        else:
            self.pause_btn.setText("⏸ 중지")

    def next_frame(self):
        if self.cap is None or self.is_paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            self.timer.stop()
            return

        results = self.model(frame)
        annotated = results[0].plot()

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_label.setPixmap(pix)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
