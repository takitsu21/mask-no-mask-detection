
from PySide6 import QtWidgets, QtGui
from PySide6.QtGui import QIcon, QScreen
from PySide6.QtWidgets import QFileDialog, QMainWindow, QApplication, QVBoxLayout, \
    QWidget, QLineEdit, QGraphicsScene

import sys
from src.KerasTrain import KerasTrain
import ctypes
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QListWidget, QListWidgetItem
from PySide6.QtCore import QDir, QSize
from typing import Optional
import PySide6
from PySide6 import QtWidgets, QtCore, QtGui
from sys import platform
import re, time, os
from PIL import Image
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
        print(path)
        #model = KerasTrain().loadModel("model-100-epochs-3.h5")
        if path[-4:] == ".png" or path[-4:] == ".xpm" or path[-4:] == ".jpg":

            #model.detect_face_and_predict(path, f"outptut-{path}")
            # label = QtWidgets.QLabel(self)
            # label.setPixmap(QPixmap(f"outptut-{path}"))
            # label.setWindowTitle(path)
            # label.show()

            ImagePredicted("./img_tests/faces.png", None).show()
            # self.label.setPixmap(QPixmap(f"outptut-{path}"))
            # img.show()

            # frame.show()
        else:
            directory = QDir(path)
            #model.predictDirectory(dirPath=path)
            self.listWidget = MyListWidget(None)
            

            


            for i, file in enumerate(directory.entryList(), start=1):
                if platform == "win32" :
                    p = re.sub("/", "\\\\", f"{path}/output/{file}")
                else:
                    p = f"{path}/output/{file}"
                print(p)
                print(os.path.exists(p))
                self.listWidget.addMyItem(QListWidgetItem(QIcon(p), f"image-{i}"), p)
            self.layout.removeWidget(self.label)
            self.layout.addWidget(self.listWidget)


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


    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        sys.exit(0)




class ImagePredicted(QtWidgets.QWidget):
    def __init__(self, path, parent: Optional[PySide6.QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent)
        self.path = path
        self.initUI()
        self.show()

    def initUI(self):
        hbox = QVBoxLayout(self)
        pixmap = QPixmap(self.path)

        lbl = QtWidgets.QLabel(self)
        lbl.setPixmap(pixmap)

        hbox.addWidget(lbl)
        self.setLayout(hbox)

        self.move(300, 200)
        self.setWindowTitle('Image with PyQt')

class imgWidget(QtWidgets.QWidget):
    def __init__(self,path, parent: Optional[PySide6.QtWidgets.QWidget] = ...) -> None:
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
        primaryScreenSize = QScreen.availableGeometry(QApplication.primaryScreen())
        width, height = primaryScreenSize.width(), primaryScreenSize.height()
        newWidth, newHeight = im.size

        newWidth = min(newWidth, width)
        newHeight = min(newHeight, height)
        self.resize(newWidth, newHeight)
        self.setFixedSize(self.size())

        # self.resize(500,500)
        self.frame = None
        self.title= name
        
        print(self.fPath)
        print(os.path.exists(self.fPath))
        self.setWindowTitle(self.title)
        self.frame = imgWidget(self.fPath,self)
        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout(self)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.layout.addWidget(self.frame)
       
        
        # self.show()


class MyListWidget(QListWidget):

    def __init__(self, parent: Optional[PySide6.QtWidgets.QWidget] = ...) -> None:
        """
        Custom QListWidget to display every image from a folder
        :param parent: parent widget
        """
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(50, 50))
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
        
        # try:
        super().mouseDoubleClickEvent(event)
        name = self.selectedItems()[0].text()
        print("name",name)
        print("fPath",self.paths[name])
        frame = FrameImage(self.paths[name], name, self)
        frame.show()
        # except:
        #     pass



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