
import sys
import cv2
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox, QProgressDialog,
    QMainWindow, QAction, QMenuBar, QTextEdit, QDialog, QFrame,
    QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt, QProcess
from ultralytics import YOLO

from data_integrator import DataIntegrator
from training_settings_dialog import TrainingSettingsDialog


class WideMenuBar(QMenuBar):
    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(1000)
        return size

class LogWindow(QDialog):
    def __init__(self, process=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process = process
        self.force_close = False

        self.setWindowTitle("YOLO 학습 로그")
        self.resize(800, 600)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def append_text(self, text):
        self.text_edit.append(text)
        self.text_edit.ensureCursorVisible()
    
    def closeEvent(self, event):
        if self.force_close:
            self.force_close = False
            event.accept()
            return

        reply = QMessageBox.question(
            self, "학습 중단 확인",
            "정말 닫으시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.process and self.process.state() == QProcess.Running:
                print("Kill yolo process!!")
                self.process.kill()
            event.accept()
        else:
            event.ignore()


class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO ADE - Auto Data Embedding")
        self.setWindowIcon(QIcon("icons/icon.ico"))
        self.resize(900, 700)

        self.training_dialog = TrainingSettingsDialog(self)

        # 메뉴바 생성
        menu_bar = WideMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu("파일")
        open_action = QAction("영상 열기", self)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)

        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menu_bar.addMenu("도움말")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black; border-radius: 5px")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.open_btn = QPushButton("영상 로드")
        self.pause_btn = QPushButton("▶ 재생")
        self.embed_btn = QPushButton("임베딩 시작")
        self.integ_btn = QPushButton("데이터 통합")
        self.train_btn = QPushButton("YOLO 학습하기")

        btn_css = "height: 20px"
        self.open_btn.setStyleSheet(btn_css)
        self.pause_btn.setStyleSheet(btn_css)
        self.embed_btn.setStyleSheet(btn_css)
        self.integ_btn.setStyleSheet(btn_css)
        self.train_btn.setStyleSheet(btn_css)

        self.open_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.embed_btn.setEnabled(False)

        self.open_btn.clicked.connect(self.open_video)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.embed_btn.clicked.connect(self.start_embedding)
        self.integ_btn.clicked.connect(self.open_integrating_dialog)
        self.train_btn.clicked.connect(self.open_training_dialog)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.embed_btn)
        btn_layout.addWidget(self.integ_btn)


        self.model_path = ""    # 영상 추론, 데이터 임베딩 시 사용하는 model 경로
        self.class_map = {}


        # 세로 구분선 추가
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        btn_layout.addWidget(separator)

        # 학습하기 버튼 추가
        btn_layout.addWidget(self.train_btn)

        # 가로 구분선
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)   # 수평선 (Horizontal Line)
        hline.setFrameShadow(QFrame.Sunken) # 살짝 입체감 (옵션)
        hline.setStyleSheet("color: #0f0f0f; background-color: gray; height: 1px; margin-top: 0px; margin-bottom: 0px;")
        hline.setFixedHeight(1)

        # 모델 선택 버튼
        model_layout = QHBoxLayout()
        self.model_btn = QPushButton("📁 모델 경로")
        self.model_btn.setStyleSheet(btn_css)
        self.model_btn.clicked.connect(self.open_model)
        # 모델 경로 표시 레이블
        self.cur_model = QLabel(f"{self.model_path}")
        self.cur_model.setStyleSheet("font-size: 12px; font-weight: bold;")
        self.cur_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # 모델 드롭 버튼
        self.model_drop_btn = QPushButton("✕")
        self.model_drop_btn.setEnabled(False)
        self.model_drop_btn.hide()
        self.model_drop_btn.clicked.connect(self.drop_model)
        self.model_drop_btn.setFixedSize(22, 22)
        self.model_drop_btn.setStyleSheet("""
            QPushButton {
                color: black;
                font-size: 14px;
                background-color: #fdfdfd
                border: 1px solid #d1d1d1;
                transition: background-color 0.5s ease, border-color 0.5s ease
            }
             QPushButton:hover {
                background-color: #e0eef9;
                border: 1px solid #52a4e2;
            }
        """)
        # 모델 레이아웃에 위젯 붙이기
        model_layout.addWidget(self.model_btn)
        model_layout.addWidget(self.cur_model)
        model_layout.addWidget(self.model_drop_btn)


        # 레이아웃 병합
        layout = QVBoxLayout()
        layout.addLayout(btn_layout)
        layout.addWidget(hline)
        layout.addLayout(model_layout)
        layout.addWidget(self.video_label, 1)

        central_widget.setLayout(layout)

        self.model = YOLO(self.model_path, task='detect')
        self.cap = None
        self.video_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_paused = False

        self.log_window = None
        self.log_buffer = []
        self.epoch_end_reached = False
    

    def open_integrating_dialog(self):
        dialog = DataIntegrator(self)
        dialog.exec_()

    def open_training_dialog(self):
        dialog = TrainingSettingsDialog(self)
        dialog.exec_()
    
    def show_about(self):
        QMessageBox.about(self, "Yolo ADE",
                        "\nVersion: 1.0.2\n"
                        "Copyright © 2025 ohsopp. All rights reserved.")


    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "MP4 files (*.mp4)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.video_path = path
            self.is_paused = False
            self.pause_btn.setEnabled(True)
            self.embed_btn.setEnabled(True)
            self.pause_btn.setText("⏸ 중지")
            self.timer.start(30)
    
    def open_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "Pytorch models (*.pt)")
        if path:
            self.model_path = path
            self.cur_model.setText(self.model_path)
            self.model = YOLO(self.model_path, task='detect')
            self.class_map = self.model.names
            print(self.class_map)
            self.open_btn.setEnabled(True)
            self.model_drop_btn.setEnabled(True)
            self.model_drop_btn.show()
    
    def drop_model(self):
        self.model_path = ""
        self.class_map = {}
        self.cur_model.setText(self.model_path)
        if self.model is not None:
            del self.model  # 명시적으로 모델 드롭
            self.model = None
        self.model_drop_btn.setEnabled(False)   # 다시 드롭 버튼 비활성화
        self.model_drop_btn.hide()
        self.open_btn.setEnabled(False)
        self.pause_btn.setText("▶ 재생")
        self.pause_btn.setEnabled(False)
        self.embed_btn.setEnabled(False)
        # 1. 영상 캡처 객체 해제(열려 있다면)
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        # 2. 타이머 정지 (영상 프레임 업데이트 중이라면)
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        # 3. 화면 검정색으로 초기화
        black_pixmap = QPixmap(self.video_label.width(), self.video_label.height())
        black_pixmap.fill(Qt.black)
        self.video_label.setPixmap(black_pixmap)
        # 4. 상태 변수 초기화
        self.video_path = None
        self.is_paused = False
        print("Model Dropped.")


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

        results = self.model(frame, verbose=False)
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
        
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        image_dir = os.path.join(base_dir, f'dataset_{video_name}/images')
        label_dir = os.path.join(base_dir, f'dataset_{video_name}/labels')
        img_bbox_dir = os.path.join(base_dir, f'dataset_{video_name}/img_bbox')

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_bbox_dir, exist_ok=True)


        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = 2

        progress = QProgressDialog("임베딩 중...", "취소", 0, total_frames // frame_interval, self)
        progress.setWindowTitle("진행 중")
        progress.setWindowModality(Qt.WindowModal)
        progress.resize(400, 150)
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
                results = self.model(frame)[0]
                detected_boxes = [box for box in results.boxes if int(box.cls.item()) in self.class_map]

                if not detected_boxes:
                    frame_idx += 1
                    continue

                frame_name = f"{saved_frame_idx:06d}"
                frame_path = os.path.join(image_dir, frame_name + '.jpg')
                label_path = os.path.join(label_dir, frame_name + '.txt')
                result_img_path = os.path.join(img_bbox_dir, frame_name + '.jpg')

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
                        cv2.putText(annotated_frame, self.class_map[cls_id], (x1, y1 - 10),
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
    

    def training_finished(self):
        reply = QMessageBox.information(
            self.log_window,  # 부모 창으로 설정
            "학습 완료",
            "모델 학습이 완료되었습니다.",
            QMessageBox.Ok
        )
        if reply == QMessageBox.Ok:
            self.log_window.force_close = True
            self.log_window.close()
    
    def handle_process_finished(self, exitCode, exitStatus):
        if exitStatus == QProcess.NormalExit and exitCode == 0:
            QMessageBox.information(None, "완료", "모델 학습이 완료되었습니다.")
            print("학습 완료 후 정상적으로 종료됨")
            self.log_window.force_close = True
            self.log_window.close()
        else:
            print("학습이 강제 종료되었거나 오류로 종료됨")
    

    def start_training(self, epoch, batch, yaml_file):

        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)

        self.epoch = epoch
        self.batch = batch
        self.yaml_file = yaml_file


        # YOLO 학습 명령어 생성
        command = [
            "yolo", "task=detect", "mode=train",
            "model=yolo11n.pt",  # 필요시 변경
            f"data={yaml_file}",
            f"epochs={epoch}",
            "imgsz=640",
            f"batch={batch}"
        ]

        # 명령 실행
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        # 로그창 생성성
        self.log_window = LogWindow(process=self.process)
        self.log_window.setFont(font)
        self.log_window.show()

        try:
            self.process.readyReadStandardOutput.disconnect()
        except Exception:
            pass

        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.handle_process_finished)


        # 개발할 때, 빌드할 때 모두 가능하도록 절대 경로 + 리스트형 인자
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        yolo_path = os.path.join(base_path, "yolo.exe")

        command = [
            yolo_path,
            "task=detect", "mode=train",
            "model=yolo11n.pt",
            f"data={self.yaml_file}",
            f"epochs={self.epoch}",
            "imgsz=640",
            f"batch={self.batch}"
        ]

        self.process.start(command[0], command[1:])

        self.header_printed = False
        self.init_log_shown = False
        self.epoch_end_reached = False
        self.log_buffer = []



    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        lines = data.splitlines()

        epoch = self.epoch
        batch = self.batch
        yaml_file = self.yaml_file

        if not hasattr(self, "header_printed"):
            self.header_printed = False

        if not hasattr(self, "init_log_shown"):
            self.init_log_shown = False


        # 최초 출력 전 '학습 준비 중...' 출력
        if not self.header_printed and not self.init_log_shown:
            self.log_window.append_text("학습 준비 중...")
            self.init_log_shown = True

        for line in lines:
            if "100%|" in line:
                line = line.replace("#", "█")

            if "100%|" in line and "Class" not in line and "all" not in line:
                self.epoch_end_reached = True
                self.log_buffer = [line.strip() + "\n"]

            elif self.epoch_end_reached:
                if "class" in line.lower():
                    self.log_buffer = [l for l in self.log_buffer if "class" not in l.lower()]
                    self.log_buffer.append(line.strip())
                else:
                    self.log_buffer.append(line.strip())

                if "all" in line.lower():
                    try:
                        epoch_line = next((l for l in self.log_buffer if "100%|" in l and "all" not in l and "Class" not in l), "")
                        all_line = next((l for l in self.log_buffer if l.lower().startswith("all")), "")

                        epoch_parts = epoch_line.split()
                        all_parts = all_line.split()

                        log_epoch = epoch_parts[0]
                        gpu_mem = epoch_parts[1]
                        box_loss = epoch_parts[2]
                        cls_loss = epoch_parts[3]
                        dfl_loss = epoch_parts[4]
                        mAP50 = all_parts[5]
                        mAP50_95 = all_parts[6]

                        if not self.header_printed:
                            header_html = (
                                '<pre style="background-color: #f0f0f0; font-weight: bold; margin:0;">'
                                f"{'Epoch':>10} {'GPU_mem':>12} {'box_loss':>12} {'cls_loss':>12} "
                                f"{'dfl_loss':>12} {'mAP50':>12} {'mAP50-95':>12}"
                                '</pre>'
                            )
                            self.log_window.text_edit.clear()
                            self.log_window.text_edit.setText(f"\n- Setting Info -\nEpoch num  : {epoch}\nBatch Size : {batch}\n.yaml path : {yaml_file}\n")
                            self.log_window.append_text(header_html)
                            self.header_printed = True

                        values_html = (
                            '<pre style="margin:0;">'
                            f"{log_epoch:>10} {gpu_mem:>12} {box_loss:>12} {cls_loss:>12} "
                            f"{dfl_loss:>12} {mAP50:>12} {mAP50_95:>12}"
                            '</pre>'
                        )
                        separator_html = (
                            '<pre style="margin:0; color: #d3d3d3;">' + '-' * 88 + '</pre>'
                        )

                        self.log_window.append_text(values_html)
                        self.log_window.append_text(separator_html)

                    except (IndexError, StopIteration):
                        pass

                    self.log_buffer = []
                    self.epoch_end_reached = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
