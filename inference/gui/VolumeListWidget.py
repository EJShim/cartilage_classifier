from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class E_VolumeListWidget(QListWidget):
    def __init__(self, parent = None):
        super(QListWidget, self).__init__(parent)

        self.mainFrm = parent
        #self.setAlternatingRowColors(True)

        self.itemDoubleClicked.connect(self.onRenderItem)

    def onRenderItem(self, item):
        idx = self.row(item)

        self.mainFrm.Mgr.RenderPreProcessedObject(idx)
        self.mainFrm.Mgr.Redraw()
        self.mainFrm.Mgr.Redraw2D()
