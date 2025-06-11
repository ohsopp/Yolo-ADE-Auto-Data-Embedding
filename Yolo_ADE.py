
import sys
import cv2
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox, QProgressDialog,
    QMainWindow, QAction, QMenuBar, QTextEdit, QDialog,
    QLineEdit, QFormLayout, QComboBox, QFrame, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QFont
from PyQt5.QtCore import QTimer, Qt, QProcess
from ultralytics import YOLO


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


class DataIntegrator(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        self.src_dir = ""
        self.dst_dir = ""
    
    def center_on_parent(self):
        if self.parent():
            parent_geometry = self.parent().geometry()
            self_geometry = self.geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self_geometry.width()) // 2
            y = parent_geometry.y() * 2 // 3 + (parent_geometry.height() - self_geometry.height()) // 2
            self.move(x, y)


    def init_ui(self):
        self.setWindowTitle("YOLO 데이터 통합기")
        self.setFixedSize(500, 170)

        layout = QVBoxLayout()
        layout.setSpacing(5)

        warning_text = QLabel("⚠️ 소스/목적지 내의 images와 labels 폴더를 꼭 확인해주세요.")

        btn_css = "height: 20px"

        # 소스 폴더 선택
        src_layout = QHBoxLayout()
        btn_select_src = QPushButton("소스 폴더")
        btn_select_src.setStyleSheet(btn_css)
        btn_select_src.clicked.connect(self.select_src_folder)
        self.src_path_label = QLabel("선택되지 않음")
        self.src_path_label.setStyleSheet("font-weight: bold; color: #333333;")
        src_layout.addWidget(btn_select_src, 0)
        src_layout.addWidget(self.src_path_label, 1)

        # 목적지 폴더 선택
        dst_layout = QHBoxLayout()
        btn_select_dst = QPushButton("목적지 폴더")
        btn_select_dst.setStyleSheet(btn_css)
        btn_select_dst.clicked.connect(self.select_dst_folder)
        self.dst_path_label = QLabel("선택되지 않음")
        self.dst_path_label.setStyleSheet("font-weight: bold; color: #333333;")
        dst_layout.addWidget(btn_select_dst, 0)
        dst_layout.addWidget(self.dst_path_label, 1)

        # 통합 버튼
        self.btn_integrate = QPushButton("데이터 통합")
        self.btn_integrate.setStyleSheet(btn_css)
        self.btn_integrate.clicked.connect(self.integrate_data)
        self.btn_integrate.setEnabled(False)

        # 프로그래스바
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()

        layout.addWidget(warning_text)
        layout.addLayout(src_layout)
        layout.addLayout(dst_layout)
        layout.addWidget(self.btn_integrate)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)
        self.center_on_parent()

    def update_integrate_button_state(self):
        self.btn_integrate.setEnabled(bool(self.src_dir and self.dst_dir))

    def select_src_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "소스 폴더 선택")
        if folder:
            image_folder = os.path.join(folder, 'images')
            label_folder = os.path.join(folder, 'labels')
            if os.path.isdir(image_folder) and os.path.isdir(label_folder):
                self.src_dir = folder
                self.src_path_label.setText(self.src_dir)
                self.update_integrate_button_state()
            else:
                QMessageBox.warning(self, "폴더 오류", "선택한 경로 내에 'images'와 'labels' 폴더가 존재하지 않습니다.")

    def select_dst_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "목적지 폴더 선택")
        if folder:
            image_folder = os.path.join(folder, 'images')
            label_folder = os.path.join(folder, 'labels')
            if os.path.isdir(image_folder) and os.path.isdir(label_folder):
                self.dst_dir = folder
                self.dst_path_label.setText(self.dst_dir)
                self.update_integrate_button_state()
            else:
                QMessageBox.warning(self, "폴더 오류", "선택한 경로 내에 'images'와 'labels' 폴더가 존재하지 않습니다.")

    def integrate_data(self):
        if not self.src_dir or not self.dst_dir:
            QMessageBox.warning(self, "경고", "소스와 목적지 폴더를 모두 선택하세요.")
            return
        
        self.progress_bar.show()    # 시작할 때 보이기기

        image_src_dir = os.path.join(self.src_dir, 'images')
        label_src_dir = os.path.join(self.src_dir, 'labels')
        image_dst_dir = os.path.join(self.dst_dir, 'images', 'train')
        label_dst_dir = os.path.join(self.dst_dir, 'labels', 'train')

        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)

        name_prefix = 'data_'
        ext_img = '.jpg'
        ext_label = '.txt'

        def get_max_index(directory, prefix, ext):
            files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
            if not files:
                return -1
            indices = [int(f[len(prefix):-len(ext)]) for f in files]
            return max(indices)

        max_index_img = get_max_index(image_dst_dir, name_prefix, ext_img)
        max_index_label = get_max_index(label_dst_dir, name_prefix, ext_label)
        current_index = max(max_index_img, max_index_label) + 1

        image_files = sorted([f for f in os.listdir(image_src_dir) if f.endswith(ext_img)])
        label_files = sorted([f for f in os.listdir(label_src_dir) if f.endswith(ext_label)])

        if len(image_files) != len(label_files):
            QMessageBox.critical(self, "오류", "이미지와 라벨 수가 일치하지 않습니다.")
            return

        total_files = len(image_files)
        self.progress_bar.setMaximum(total_files)

        for i, (img_file, label_file) in enumerate(zip(image_files, label_files), start=1):
            new_name = f"{name_prefix}{current_index:04d}"

            shutil.copy(
                os.path.join(image_src_dir, img_file),
                os.path.join(image_dst_dir, new_name + ext_img)
            )
            shutil.copy(
                os.path.join(label_src_dir, label_file),
                os.path.join(label_dst_dir, new_name + ext_label)
            )

            current_index += 1
            self.progress_bar.setValue(i)
            QApplication.processEvents()  # UI 업데이트
        
        
        self.progress_bar.hide()  # 완료 후 숨김

        result = QMessageBox.information(self, "완료", "✅ 데이터 통합이 완료되었습니다.")
        if result == QMessageBox.Ok:
            self.accept()



class TrainingSettingsDialog(QDialog):
    def __init__(self, yolo_app, parent=None):
        super().__init__(parent)
        self.yolo_app = yolo_app  # YoloApp 인스턴스 저장

        self.setWindowTitle("YOLO 학습 설정")
        self.resize(300, 120)

        # 에포크 입력 (숫자만 가능)
        self.epoch_input = QLineEdit()
        self.epoch_input.setValidator(QIntValidator(1, 1000, self))
        self.epoch_input.setPlaceholderText("예: 50")
        self.epoch_input.textChanged.connect(self.update_train_button_state)

        # 배치 크기 선택 (8 ~ 512)
        self.batch_combo = QComboBox()
        self.batch_sizes = [2 ** i for i in range(3, 10)]  # 8 ~ 512
        for size in self.batch_sizes:
            self.batch_combo.addItem(str(size))

        # YAML 선택
        self.yaml_path = ""
        self.yaml_button = QPushButton(".yaml 파일 선택")
        self.yaml_label = QLabel("선택된 파일 없음")
        self.yaml_button.clicked.connect(self.select_yaml_file)

        # 학습 버튼
        self.train_button = QPushButton("학습하기")
        self.train_button.setEnabled(False)  # 초기에는 비활성화
        self.train_button.clicked.connect(self.on_train_clicked)

        # 폼 레이아웃
        form_layout = QFormLayout()
        form_layout.addRow("에포크 수:", self.epoch_input)
        form_layout.addRow("배치 크기:", self.batch_combo)

        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(self.yaml_button)
        yaml_layout.addWidget(self.yaml_label)

        # 전체 레이아웃
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(yaml_layout)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def on_train_clicked(self):
        epoch = int(self.epoch_input.text())
        batch = int(self.batch_combo.currentText())
        yaml_file = self.yaml_path

        self.yolo_app.start_training(epoch, batch, yaml_file)
        self.accept()


    def update_train_button_state(self):
        # 에포크와 yaml 파일이 모두 입력되었을 때만 버튼 활성화
        epoch_valid = bool(self.epoch_input.text().strip())
        yaml_valid = bool(self.yaml_path)
        self.train_button.setEnabled(epoch_valid and yaml_valid)

    def select_yaml_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "YAML 파일 선택", "", "YAML Files (*.yaml)")
        if file_name:
            self.yaml_path = file_name
            self.yaml_label.setText(file_name.split("/")[-1])
            self.update_train_button_state()



class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO ADE - Auto Data Embedding")
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
        about_action = QAction("정보", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black; border-radius: 5px")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.open_btn = QPushButton("영상 열기")
        self.pause_btn = QPushButton("⏸ 중지")
        self.embed_btn = QPushButton("임베딩 시작")
        self.integ_btn = QPushButton("데이터 통합")
        self.train_btn = QPushButton("YOLO 학습하기")

        btn_css = "height: 20px"
        self.open_btn.setStyleSheet(btn_css)
        self.pause_btn.setStyleSheet(btn_css)
        self.embed_btn.setStyleSheet(btn_css)
        self.integ_btn.setStyleSheet(btn_css)
        self.train_btn.setStyleSheet(btn_css)

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


        model_path = "runs/detect/train7/weights/best.pt"


        # 세로 구분선 추가
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        btn_layout.addWidget(separator)

        btn_layout.addWidget(self.train_btn)

        cur_model = QLabel(f"모델 경로 : {model_path}")
        cur_model.setStyleSheet("font-size: 12px; font-weight: bold;")
        cur_model.adjustSize()


        layout = QVBoxLayout()
        layout.addLayout(btn_layout)
        layout.addWidget(cur_model, 0)
        layout.addWidget(self.video_label, 1)

        central_widget.setLayout(layout)

        self.model = YOLO(model_path)
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
        QMessageBox.information(self, "정보", "\n버전 1.0.0\n")

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
        
        video_name = os.path.splitext(self.video_path)[0]

        image_dir = os.path.join(base_dir, f'dataset_{video_name}/images')
        label_dir = os.path.join(base_dir, f'dataset_{video_name}/labels')
        img_bbox_dir = os.path.join(base_dir, f'dataset_{video_name}/img_bbox')

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_bbox_dir, exist_ok=True)

        model = self.model      # 추론 모델과 임베딩 모델 동기화

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = 2
        class_map = {0: 'putter', 1: 'ball'}    # 모델의 클래스와 맞추기!!

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
                results = model(frame)[0]
                detected_boxes = [box for box in results.boxes if int(box.cls.item()) in class_map]

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

        self.process.start(" ".join(command), QProcess.ReadOnly)

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
                            self.log_window.text_edit.setText(f"\n- Setting Info -\nEpoch : {epoch}\nBatch Size : {batch}\n.yaml path : {yaml_file}\n")
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
