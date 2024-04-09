from PyQt6 import QtWidgets
from view import Ui_MainWindow
from event import Event


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    
    eventButton = Event(ui)

    ui.addImage.clicked.connect(eventButton.addImage)
    ui.turnLeft.clicked.connect(eventButton.turnLeft)
    ui.turnRight.clicked.connect(eventButton.turnRight)
    ui.removeText.clicked.connect(eventButton.removeText)
    ui.pasteImage.clicked.connect(eventButton.pasteImage)
    ui.zoomIn.clicked.connect(eventButton.zoomIn)
    ui.zoomOut.clicked.connect(eventButton.zoomOut)
    ui.actualSize.clicked.connect(eventButton.actualSize)
    ui.findText.clicked.connect(eventButton.findText)
    ui.fitImage.clicked.connect(eventButton.fitImage)
    ui.scanImage.clicked.connect(eventButton.ScanIMG)
    ui.savePDF.clicked.connect(eventButton.savePDF)

    # Chuyển trạng thái các button liên quan đến xử lý ảnh qua trạng thái không bấm được
    ui.turnLeft.setEnabled(False)
    ui.turnRight.setEnabled(False)
    ui.zoomIn.setEnabled(False)
    ui.zoomOut.setEnabled(False)
    ui.actualSize.setEnabled(False)
    ui.fitImage.setEnabled(False)
    ui.scanImage.setEnabled(False)     
    MainWindow.show()
    sys.exit(app.exec())
