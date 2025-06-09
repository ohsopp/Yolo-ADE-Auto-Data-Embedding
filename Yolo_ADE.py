import sys
import cv2
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox, QProgressDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO ADE - Auto Data Embedding")
        self.resize(800, 600)

        self.model = YOLO("runs/detect/train5/weights/best.pt")

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: black")

        self.open_btn = QPushButton("영상 열기")
        self.pause_btn = QPushButton("⏸ 중지")
        self.embed_btn = QPushButton("임베딩 시작")
        self.pause_btn.setEnabled(False)
        self.embed_btn.setEnabled(False)

        self.open_btn.clicked.connect(self.open_video)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.embed_btn.clicked.connect(self.start_embedding)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.embed_btn)

        layout = QVBoxLayout()
        layout.addLayout(btn_layout)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        layout.addLayout(video_layout)
        self.setLayout(layout)

        self.cap = None
        self.video_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_paused = False

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "MP4 files (*.mp4);;All files (*)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.video_path = path
            self.is_paused = False
            self.pause_btn.setEnabled(True)
            self.embed_btn.setEnabled(True)
            self.pause_btn.setText("⏸ 중지")
            self.timer.start(30)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_btn.setText("▶ 재생" if self.is_paused else "⏸ 중지")

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

    def start_embedding(self):
        if not self.video_path:
            QMessageBox.warning(self, "오류", "먼저 영상을 선택하세요.")
            return

        base_dir = QFileDialog.getExistingDirectory(self, "데이터 저장 폴더 선택")
        if not base_dir:
            return

        frame_dir = os.path.join(base_dir, 'frames')
        label_dir = os.path.join(base_dir, 'labels')
        img_res_dir = os.path.join(base_dir, 'img_res')

        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_res_dir, exist_ok=True)

        model_path = "runs/detect/train5/weights/best.pt"
        model = YOLO(model_path)

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = 2
        class_map = {0: 'putter', 1: 'ball'}

        progress = QProgressDialog("임베딩 중...", "취소", 0, total_frames // frame_interval, self)
        progress.setWindowTitle("진행 중")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        frame_idx = 0
        saved_frame_idx = 0

        while cap.isOpened():
            if progress.wasCanceled():
                QMessageBox.information(self, "중단됨", "사용자에 의해 임베딩이 중단되었습니다.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                h, w, _ = frame.shape

                results = model(frame)[0]
                detected_boxes = [box for box in results.boxes if int(box.cls.item()) in class_map]

                if not detected_boxes:
                    frame_idx += 1
                    continue

                frame_name = f"{saved_frame_idx:06d}"
                frame_path = os.path.join(frame_dir, frame_name + '.jpg')
                label_path = os.path.join(label_dir, frame_name + '.txt')
                result_img_path = os.path.join(img_res_dir, frame_name + '.jpg')

                cv2.imwrite(frame_path, frame)
                annotated_frame = frame.copy()

                with open(label_path, 'w') as f:
                    for box in detected_boxes:
                        cls_id = int(box.cls.item())
                        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h

                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, class_map[cls_id], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imwrite(result_img_path, annotated_frame)
                print(f"✅ {frame_name}.jpg 저장 완료")

                saved_frame_idx += 1
                progress.setValue(saved_frame_idx)

            frame_idx += 1

        cap.release()
        embedding_canceled = progress.wasCanceled()
        progress.close()

        if not embedding_canceled:
            QMessageBox.information(self, "임베딩 완료", "🎉 모든 임베딩 작업이 완료되었습니다.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
