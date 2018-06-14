from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class E_MainRenderingWidget(QWidget):
    def __init__(self, parent = None):
        super(E_MainRenderingWidget, self).__init__(parent)        

        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)
        self.sliceView = None


        mainRenderWidget = QWidget()
        self.mainLayout.addWidget(mainRenderWidget)
        self.mainRenderLayout = QVBoxLayout()
        mainRenderWidget.setLayout(self.mainRenderLayout)


        sliceRenderWidget = QWidget()
        self.mainLayout.addWidget(sliceRenderWidget)
        self.sliceRenderLayout = QVBoxLayout()
        sliceRenderWidget.setLayout(self.sliceRenderLayout)



        #Set Spacing and Margins of Layouts
        self.mainLayout.setSpacing(0)
        self.mainLayout.setContentsMargins(0,0,0,0)

        self.mainRenderLayout.setSpacing(1)
        self.mainRenderLayout.setContentsMargins(0,0,1,0)

        self.sliceRenderLayout.setSpacing(1)
        self.sliceRenderLayout.setContentsMargins(0, 0, 0, 0)



        self.mainLayout.setStretch(0, 3)
        self.mainLayout.setStretch(1, 1)
    
        self.Initialize()

    def SetManager(self, Mgr):
        self.Mgr = Mgr


    def Initialize(self):
        print("Main Rendering Widget Initialized")

    def AddMainRenderer(self, rendererWidget):        
        self.mainRenderLayout.addWidget(rendererWidget)      
    
    def AddSliceRenderer(self, rendererWidget):
        self.sliceRenderLayout.addWidget(rendererWidget)  
        
        if self.sliceView == None:
            self.sliceView = rendererWidget




    def SetViewMainView(self):
        self.sliceView.setParent(self.sliceRenderLayout.parentWidget())
        self.sliceRenderLayout.insertWidget(0,self.sliceView)
        self.mainLayout.setStretch(0, 3)
        self.mainLayout.setStretch(1, 1)
        


    def SetViewGridView(self):
        self.sliceView.setParent(self.mainRenderLayout.parentWidget())
        self.mainRenderLayout.insertWidget(0,self.sliceView)
        self.mainLayout.setStretch(0, 1)
        self.mainLayout.setStretch(1, 1)

        