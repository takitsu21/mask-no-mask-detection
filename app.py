
from PySide6 import QtWidgets, QtGui
from PySide6.QtGui import QIcon, QScreen
from PySide6.QtWidgets import QFileDialog, QMainWindow, QApplication, QVBoxLayout, \
    QWidget

import sys
from src.KerasTrain import KerasTrain
import ctypes
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QListWidget, QListWidgetItem
from PySide6.QtCore import QDir, QSize
from typing import Optional
import PySide6
from PySide6 import QtWidgets, QtCore, QtGui
import os
from PIL import Image
from PySide6.QtGui import QAction


def find_model(extension: str = ".h5", path: str = "."):
    for file in os.listdir(path):
        if file.endswith(extension):
            return file


class MultiView(QtWidgets.QWidget):

    def __init__(self, parent: Optional[QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent=parent)
        self.setAcceptDrops(True)
        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(self)
        self.pixmap = QPixmap("./ressources/assets/DRAGNDROP.png")
        self.label.setPixmap(self.pixmap)
        self.layout.addWidget(self.label, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(self.layout)
        self.listWidget = None
        self.modelPath = find_model()

    def loadPredictions(self, path):
        model = KerasTrain().loadModel(self.modelPath)
        if path[-4:] == ".png" or path[-4:] == ".xpm" or path[-4:] == ".jpg":
            split_f = path.split("/")[-1].split(".")
            output_dir = "output"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            f_output = f"{output_dir}/outptut-{split_f[0]}.{split_f[1]}"
            model.detectFaceAndPredict(path, f_output)
            frame = FrameImage(f_output, path, self)
            frame.show()
        else:

            model.predictDirectory(dirPath=path)
            output_dir = "output"
            directory = QDir(output_dir)

            self.listWidget = MyListWidget(None)

            for i, file in enumerate(directory.entryList(), start=1):

                p = f"{output_dir}/{file}"

                if not os.path.isdir(p):
                    self.listWidget.addMyItem(
                        QListWidgetItem(QIcon(p), f"image-{i}"), p)
            self.layout.removeWidget(self.label)
            self.layout.addWidget(self.listWidget)

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
        self.loadPredictions(path)


class MenuBar(QtWidgets.QMenuBar):

    def __init__(self, parent: Optional[QtWidgets.QWidget] = ...) -> None:
        """
        MenuBar of every window
        :param bopen: True if main window application opened, otherwise False
        :param parent: parent widget
        """
        super().__init__()

        self.parent = parent
        self.frame = parent.frame
        self.fileMenu = self.addMenu("File")

        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)
        self.importModel = QAction("Import model", self)
        self.importModel.setShortcut("Ctrl+i")
        self.importModel.triggered.connect(self.newModelPath)

        self.open = self.fileMenu.addMenu("Open")

        self.openFile = QAction("Open file", self)
        self.openFile.setShortcut("Ctrl+o")
        self.openFile.triggered.connect(self.loadFile)
        self.open.addAction(self.openFile)

        self.openFold = QAction("Open folder", self)
        self.openFold.setShortcut("Ctrl+Shift+O")
        self.openFold.triggered.connect(self.loadFolder)
        self.open.addAction(self.openFold)

        self.fileMenu.addAction(self.importModel)
        # self.fileMenu.addAction(self.openFile)
        # self.fileMenu.addAction(self.openFold)

        self.layout.addWidget(self.fileMenu)
        self.setLayout(self.layout)

    def newModelPath(self):
        """
        Open a file dialog to choose the model path
        """
        path = QFileDialog.getOpenFileName(
            self.parent, "Open file", "", "Model files (*.h5)")[0]
        if path != "":
            self.frame.modelPath = path

    def loadFolder(self):
        """
        Open a file dialog to choose the folder path
        """
        path = QFileDialog.getExistingDirectory(
            self.parent, "Open folder", "")[0]
        if path != "":
            self.frame.loadPredictions(path)

    def loadFile(self):
        """
        Open a file dialog to choose the folder path
        """
        path = QFileDialog.getOpenFileName(self)[0]
        if path != "":
            self.frame.loadPredictions(path)


class PredictorVisualizer(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Detector")
        self.resize(1280, 720)
        self.frame = MultiView(self)

        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.layout.addWidget(self.frame)
        self.menu = MenuBar(self)
        self.layout.setMenuBar(self.menu)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        sys.exit(0)


class ImgWidget(QtWidgets.QWidget):
    def __init__(self, path, parent: Optional[PySide6.QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent)
        self.path = path
        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(self)
        self.pixmap = QPixmap(path)
        self.label.setPixmap(self.pixmap)
        self.layout.addWidget(self.label, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(self.layout)
        self.listWidget = None


class FrameImage(QMainWindow):
    def __init__(self, fPath, name, parent: Optional[QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent=parent)
        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)
        self.fPath = fPath
        im: Image = Image.open(self.fPath)
        primaryScreenSize = QScreen.availableGeometry(
            QApplication.primaryScreen())
        width, height = primaryScreenSize.width(), primaryScreenSize.height()
        newWidth, newHeight = im.size
        newWidth, newHeight = newWidth+50, newHeight+50
        newWidth = min(newWidth, width)
        newHeight = min(newHeight, height)
        self.resize(newWidth, newHeight)
        # self.setFixedSize(self.size())

        # self.resize(500,500)
        self.frame = None
        self.title = name

        print(self.fPath)
        print(os.path.exists(self.fPath))
        self.setWindowTitle(self.title)
        self.frame = ImgWidget(self.fPath, self)
        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.layout.addWidget(self.frame)


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
        super().mouseDoubleClickEvent(event)
        name = self.selectedItems()[0].text()
        frame = FrameImage(self.paths[name], name, self)
        frame.show()


if __name__ == "__main__":
    """
    run the program
    """
    if sys.platform == "win32":
        appId = 'PredictorVisualizer'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appId)
    app = QApplication(sys.argv)
    with open("ressources/styles/dark-theme.qss") as f:
        lines = " ".join(f.readlines())
    app.setStyleSheet(lines)
    app.setWindowIcon(QIcon("ressources/assets/app-logo.png"))
    app.setDesktopFileName("ImageAnnotator")

    w = PredictorVisualizer()
    w.show()
    sys.exit(app.exec())
