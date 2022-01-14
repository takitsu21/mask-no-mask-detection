
from PySide6 import QtWidgets, QtGui
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog, QMainWindow, QApplication, QVBoxLayout, \
    QWidget, QLineEdit

import sys
from src.KerasTrain import KerasTrain
import ctypes
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QListWidget, QListWidgetItem
from PySide6.QtCore import QDir, QSize
from typing import Optional
import PySide6
from PySide6 import QtWidgets, QtCore, QtGui

class PredictorVisualize(QMainWindow):
    def __init__(self) -> None:
        super().__init__()


        self.setWindowTitle("Image Detector")
        self.resize(1280, 720)

        self.label = QtWidgets.QLabel(self)
        self.pixmap = QPixmap("./ressources/assets/DRAGNDROP.png")
        self.label.setPixmap(self.pixmap)
        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.label.layout.addWidget(self.label, alignment=QtCore.Qt.AlignCenter)
        # self.layout.addWidget(self.label, alignment=QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.edit = QLineEdit()
        self.edit.setDragEnabled(True)
        self.setAcceptDrops(True)





    def dragEnterEvent(self, e):
        """
        Check whether we enter the drag zone event
        """
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        Check when we drop something in the drag zone
        """
        url = e.mimeData().urls()[0]
        path = url.toLocalFile()
        model = KerasTrain().loadModel("model-100-epochs-3.h5")
        if path[-4:] == ".png" or path[-4:] == ".xpm" or path[-4:] == ".jpg":

            model.detect_face_and_predict(path, f"outptut-{path}")

            # frame.show()
        else:
            directory = QDir(path)
            model.predictDirectory(dirPath=path)
            self.listWidget = MyListWidget()

            for i, file in enumerate(directory.entryList(), start=1):
                self.listWidget.addMyItem(QListWidgetItem(QIcon(f"{path}/output/{file}"), f"image-{i}"), f"{path}/{file}")

            self.layout.addWidget(self.listWidget)


class MyListWidget(QListWidget):

    def __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = ...) -> None:
        """
        Custom QListWidget to display every image from a folder
        :param parent: parent widget
        """
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(125, 125))
        self.setResizeMode(QListWidget.Adjust)
        self.paths = {}


    def addMyItem(self, item: PySide6.QtWidgets.QListWidgetItem, path: str):
        """

        :param item: contain one image
        :param path: file path
        """
        item.setSizeHint(QSize(150, 150))
        item.setTextAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)
        super().addItem(item)
        self.paths[item.text()] = path

    def mouseDoubleClickEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
        """
        Open image by double clicking mouse event
        """
        try:
            super().mouseDoubleClickEvent(event)
            name = self.selectedItems()[0].text()
            label = QtWidgets.QLabel()
            label.setPixmap(QPixmap(self.paths[name]))
            label.setWindowTitle(name)
            label.show()

        except:
            pass




if __name__ == "__main__":
    """
    run the program
    """
    if sys.platform == "win32":
        appId = 'PredictorVisualize'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appId)
    app = QApplication(sys.argv)
    with open("ressources/styles/dark-theme.qss") as f:
        lines = " ".join(f.readlines())
    app.setStyleSheet(lines)
    app.setWindowIcon(QIcon("ressources/assets/app-logo.png"))
    app.setDesktopFileName("ImageAnnotator")

    w = PredictorVisualize()
    w.show()
    sys.exit(app.exec())