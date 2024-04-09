import os
import convert
import read

from cv2 import QT_FONT_BLACK

from PyQt6.QtWidgets import QFileDialog, QApplication, QInputDialog
from PyQt6.QtGui import QPixmap, QTransform, QTextCharFormat, QBrush, QColor
from PyQt6.QtCore import QSize
from PIL import Image

class Event:
    def __init__(self, ui):
        self.ui = ui
        self.image_path = ''  # khởi tạo biến lưu đường dẫn ảnh
        self.setMaxMinSizes()
        self.rotateL = 0
        self.rotateR = 0

    def setMaxMinSizes(self):
        label_size = self.ui.imageLabel.size()
        max_width = int(label_size.width() * 4)
        max_height = int(label_size.height() * 4)
        min_width = int(label_size.width() * 0.4)
        min_height = int(label_size.height() * 0.4)
        self.max_width = max_width
        self.max_height = max_height
        self.min_width = min_width
        self.min_height = min_height
            
    def addImage(self):
        # Thiết lập thư mục mặc định là thư mục của file ảnh trước đó được chọn
        folder = os.path.dirname(self.image_path) if self.image_path else ''
        image_name, _ = QFileDialog.getOpenFileName(None, "", folder, "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_name:
            # Load ảnh vào QPixmap
            pixmap = QPixmap(image_name)
            #Load ảnh vào label
            self.ui.imageLabel.setPixmap(pixmap)
            # lưu đường dẫn ảnh mới được chọn
            self.image_path = image_name
            # Lưu kích thước gốc của ảnh
            self.original_size = pixmap.size()
            # Chuyển trạng thái các button liên quan đến xử lý ảnh qua trạng thái bấm được
            self.ui.turnLeft.setEnabled(True)
            self.ui.turnRight.setEnabled(True)
            self.ui.zoomIn.setEnabled(True)
            self.ui.zoomOut.setEnabled(True)
            self.ui.actualSize.setEnabled(True)
            self.ui.fitImage.setEnabled(True)
            self.ui.scanImage.setEnabled(True)

            self.rotateL=0
            self.rotateR=0

    def pasteImage(self):
        try:
            clipboard = QApplication.clipboard()
            if clipboard.mimeData().hasImage():
                image = clipboard.image()
                if not image.isNull():
                    pixmap = QPixmap.fromImage(image)
                    self.ui.imageLabel.setPixmap(pixmap)
                    # Chuyển trạng thái các button liên quan đến xử lý ảnh qua trạng thái bấm được
                    self.ui.turnLeft.setEnabled(True)
                    self.ui.turnRight.setEnabled(True)
                    self.ui.zoomIn.setEnabled(True)
                    self.ui.zoomOut.setEnabled(True)
                    self.ui.actualSize.setEnabled(True)
                    self.ui.fitImage.setEnabled(True)
                    self.ui.scanImage.setEnabled(True)

                    # Lưu kích thước gốc của ảnh
                    self.original_size = pixmap.size()
        except Exception as e:
            print(e)

    def fitImage(self):
        # Lấy kích thước cửa sổ hiện tại
        window_size = self.ui.imageLabel.size()
        # Lấy kích thước ảnh hiện tại
        pixmap = self.ui.imageLabel.pixmap()
        image_size = pixmap.size()
        # Tính tỷ lệ giữa kích thước ảnh và cửa sổ hiện tại
        ratio = min(window_size.width() / image_size.width(), window_size.height() / image_size.height())
        # Scale ảnh với kích thước mới
        pixmap = self.ui.imageLabel.pixmap().scaled(int(image_size.width() * ratio), int(image_size.height() * ratio))
        # Đặt lại ảnh vào label
        self.ui.imageLabel.setPixmap(pixmap)
    
    def actualSize(self):
        pixmap = self.ui.imageLabel.pixmap().scaled(self.original_size.width(), self.original_size.height())
        self.ui.imageLabel.setPixmap(pixmap)

    def zoomOut(self):
        pixmap = self.ui.imageLabel.pixmap()
        if pixmap is not None:
            width = pixmap.width() // 2
            height = pixmap.height() // 2
            pixmap = pixmap.scaled(width, height)
            self.ui.imageLabel.setPixmap(pixmap)

    def zoomIn(self):
        pixmap = self.ui.imageLabel.pixmap()
        if pixmap is not None:
            width = pixmap.width() * 2
            height = pixmap.height() * 2
            pixmap = pixmap.scaled(width, height)
            self.ui.imageLabel.setPixmap(pixmap)

    def turnLeft(self):
        # Lấy pixmap hiện tại của label
        pixmap = self.ui.imageLabel.pixmap()

        # Xoay pixmap 90 độ sang trái
        rotated_pixmap = pixmap.transformed(QTransform().rotate(-90))

        # Set pixmap mới cho label
        self.ui.imageLabel.setPixmap(rotated_pixmap)

        self.original_size = QSize(self.original_size.height(), self.original_size.width())

        self.rotateL+=1

    def turnRight(self):
        # Lấy pixmap hiện tại của label
        pixmap = self.ui.imageLabel.pixmap()

        # Xoay pixmap 90 độ sang phải
        rotated_pixmap = pixmap.transformed(QTransform().rotate(90))

        # Set pixmap mới cho label
        self.ui.imageLabel.setPixmap(rotated_pixmap)

        self.original_size = QSize(self.original_size.height(), self.original_size.width())
        
        self.rotateR+=1
    
    def removeText(self):
        self.ui.textArea.setPlainText('')

    def findText(self):
        text_to_find, ok = QInputDialog.getText(None, "Find Text", "Enter text to find:")
        if ok:
            # Tìm kiếm từ cần tìm trong văn bản
            cursor = self.ui.textArea.document().find(text_to_find)

            # Format lại trước khi tìm kiếm
            text_format = QTextCharFormat()
            text_format.setBackground(QBrush(QColor('white')))
            previous_cursor = self.ui.textArea.textCursor()
            self.ui.textArea.selectAll()
            self.ui.textArea.mergeCurrentCharFormat(text_format)
            self.ui.textArea.setTextCursor(previous_cursor)

            while not cursor.isNull():
                # Highlight từ được tìm thấy
                text_format = QTextCharFormat()
                text_format.setBackground(QBrush(QColor('yellow')))
                cursor.mergeCharFormat(text_format)

                # Tiếp tục tìm kiếm từ tiếp theo
                cursor = self.ui.textArea.document().find(text_to_find, cursor)

    def ScanIMG(self):
        # pixmap = self.ui.imageLabel.pixmap()
        # image = pixmap.toImage()
        # image.save("./pdf/change.jpg")
        img=Image.open(self.image_path)
        rotate=0
        for i in range(0,self.rotateR):
            rotate+=90
        for i in range(0,self.rotateL):
            rotate-=90
        img=img.rotate(rotate)
        img.save("./pdf/change.png")
        str=read.Job("./pdf/change.png")
        text = self.ui.textArea.toPlainText()
        if text != '':
            str = text + "\n" + str
        self.ui.textArea.setPlainText(str)

    def savePDF(self):
        text = self.ui.textArea.toPlainText()
        file = open("./pdf/PDF.txt", "w+")
        file.write(text)
        file.close()
        fileName, _ = QFileDialog.getSaveFileName(None, "Save file", "", "PDF Files (*.pdf)")
        if fileName:
            file_name = fileName.split("/")[-1]
            print("Tên file được đặt: ", file_name)
            convert.TXTtoPDF("./pdf/PDF.txt",fileName)
        
        