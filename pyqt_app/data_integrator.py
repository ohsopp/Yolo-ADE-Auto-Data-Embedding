import os
import shutil

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QProgressBar, QApplication
)


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