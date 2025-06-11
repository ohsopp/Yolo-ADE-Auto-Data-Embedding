
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QComboBox, QFormLayout
)
from PyQt5.QtGui import QIntValidator


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