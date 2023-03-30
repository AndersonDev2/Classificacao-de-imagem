from PyQt5 import QtWidgets, QtGui, uic, QtCore
from PIL import Image
import numpy as np
import sys
import os
import config
import nn
import handle_data

base_dir = os.path.dirname(__file__)
os.system("cls")


class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        self.model = nn.create_model(6)
        self.model.load_weights(nn.weights_file_path)
        self.img_selected = None
        uic.loadUi(os.path.join(base_dir, "gui.ui"), self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.CarregarImagemButton.clicked.connect(self.abrir_imagem)
        self.PredizerButton.clicked.connect(self.predizer)
        self.FecharJanela.clicked.connect(self.close)
        for button in [
                self.CarregarImagemButton, self.PredizerButton,
                self.FecharJanela,
        ]:
            button.setCursor(
                QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.posicao0 = self.pos()
        self.show()

    def mousePressEvent(self, event):
        self.posicao0 = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QtCore.QPoint(event.globalPos() - self.posicao0)
        self.move(self.x()+delta.x(), self.y()+delta.y())
        self.posicao0 = event.globalPos()

    def abrir_imagem(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir imagem", "", "")
        if file == "":
            return
        qimg = QtGui.QPixmap(file)
        qimg = qimg.scaled(520, 520)
        self.Imagem.setPixmap(qimg)
        self.img_selected = file
        return

    def predizer(self):
        if self.img_selected == None:
            return
        labels = handle_data.get_labels()
        labels_dict = {
            "buildings": "Prédios",
            "forest": "Floresta",
            "glacier": "Geleira",
            "mountain": "Montanha",
            "sea": "Mar",
            "street": "Ruas",
        }
        img = np.array([handle_data.load_image(self.img_selected)])
        model_resultado = self.model.predict(img, verbose=0)
        max_resultado = np.argmax(model_resultado)
        label_resultado = labels[max_resultado]
        resultado = labels_dict[label_resultado]
        self.Predicao.setText(resultado)
        self.img_selected = None
        return


app = QtWidgets.QApplication(sys.argv)
window = UI()
app.exec_()
