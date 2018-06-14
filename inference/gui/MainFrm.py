#-*- encoding: utf8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from gui.VolumeRenderingWidget import E_VolumeRenderingWidget
from gui.VolumeListWidget import E_VolumeListWidget
from gui.VolumeTreeWidget import E_VolumeTreeWidget
from gui.RendererViewWidget import E_MainRenderingWidget
import datetime


import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys, os
from manager.Mgr import E_Manager
from manager.E_Threads import *


import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
icon_path = os.path.join(root_path, "res", "icons")

class E_MainWindow(QMainWindow):
    update_cam = pyqtSignal()


    def __init__(self, parent = None):
        super(E_MainWindow, self).__init__(parent)

        
        self.installEventFilter(self)

        self.splash = QSplashScreen(QPixmap(os.path.join(root_path, "data", "screen.png")))
        self.splash.show()
        self.splash.finish(self)

        self.m_saveDir = '~/'
        try:
            with open(os.path.join(root_path, 'res', 'path_tmp'), 'r') as text_file:
                self.m_saveDir = text_file.read().replace('\n', '')
        except:
            with open(os.path.join(root_path, 'res', 'path_tmp'), 'w') as text_file:
                print(self.m_saveDir, file=text_file)

        self.m_capDir = '~/'
        try:
            with open(os.path.join(root_path, 'res', 'capture_path'), 'r') as text_file:
                self.m_capDir = text_file.read().replace('\n', '')
        except:
            with open(os.path.join(root_path, 'res', 'capture_path'), 'w') as text_file:
                print(self.m_capDir, file=text_file)


        self.setWindowTitle("RCT Classifier")
        self.keyPlaying = {}


        #Central Widget
        self.m_centralWidget = QWidget()
        self.setCentralWidget(self.m_centralWidget)


        #Volume List Dialog
        self.m_volumeListDlg = E_VolumeListWidget()


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.UpdateRenderer)



        #Bone Color, RCT
        self.m_bBoneColorBlack = "Black"
        self.m_bRCT = True        
        self.m_sliceSlider = [0, 0, 0]

        #vtk Renderer Widget
        self.m_vtkWidget = None
        self.m_croppingWidget = None
        self.m_vtkSliceWidget = [0,0,0]      


        #Initialize
        QApplication.processEvents()
        self.splash.showMessage("initialize gui")
        self.InitToolbar()
        self.InitCentralWidget()

        self.splash.showMessage("initialize manager")
        QApplication.processEvents() 
        self.InitManager()



        #Status Bar
        self.statusBar().showMessage('Ready')
        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(30, 40, 200, 25)        
        self.statusBar().addPermanentWidget(self.progressBar)

    def __close__(self):
        print("delete")


    def eventFilter(self, obj, event):
        # print(event)
        if event.type() == QEvent.ShortcutOverride:            
            if event.key() == Qt.Key_V:
                self.onImportVolume()
            return True # means stop event propagation
        else:
            return QMainWindow.eventFilter(self, obj, event)


    def InitToolbar(self):
        objectToolbar = QToolBar();
        objectToolbar.setIconSize(QSize(50, 50))        
        objectToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        objectToolbar.setMovable(False)
        self.addToolBar(Qt.RightToolBarArea, objectToolbar)
        # mainTab.addTab(objectToolbar, "3D Objects")        


        self.volumeWidget = E_VolumeRenderingWidget()
        objectToolbar.addWidget(self.volumeWidget)

        objectToolbar.addSeparator()


        checkWidget = QWidget()
        objectToolbar.addWidget(checkWidget)
        checkLayout = QVBoxLayout()
        checkLayout.setSpacing(0)
        
        checkWidget.setLayout(checkLayout)

        #Show/hide Volume List
        treeViewCheck = QCheckBox("Volume Tree")
        treeViewCheck.setCheckState(2)
        treeViewCheck.stateChanged.connect(self.onVolumeTreeState)
        checkLayout.addWidget(treeViewCheck)

        croppingViewCheck = QCheckBox("Cropping View")
        croppingViewCheck.setCheckState(2)
        croppingViewCheck.stateChanged.connect(self.onCroppingViewState)
        checkLayout.addWidget(croppingViewCheck)


        self.classCheck = QCheckBox("CAM")
        self.classCheck.setCheckState(2)
        self.classCheck.setEnabled(False)
        self.classCheck.stateChanged.connect(self.onClassActivationMapState)
        checkLayout.addWidget(self.classCheck)
    
        objectToolbar.addSeparator()

        ##View 1, 4 View
        viewcontrolLayout = QVBoxLayout()
        viewControl = QGroupBox("View Control")
        viewControl.setLayout(viewcontrolLayout)
        radioNormal = QRadioButton("Normal View")
        radioNormal.clicked.connect(self.SetViewModeNromal)
        radioGrid = QRadioButton("Grid View")
        radioGrid.clicked.connect(self.SetViewModeGrid)


        viewcontrolLayout.addWidget(radioNormal)
        viewcontrolLayout.addWidget(radioGrid)                   
        viewcontrolLayout.itemAt(0).widget().setChecked(True)        
        objectToolbar.addWidget(viewControl)        
        
        objectToolbar.addSeparator()        
        objectToolbar.addSeparator()        

        networkToolbar = QToolBar();
        networkToolbar.setIconSize(QSize(50, 50))
        networkToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        networkToolbar.setMovable(False)
        self.addToolBar(networkToolbar)
        # mainTab.addTab(networkToolbar, "VRN")


        #Import Volume addAction
        volumeAction = QAction(QIcon(icon_path + "/051-document.png"), "Import Volume", self)
        volumeAction.triggered.connect(self.onImportVolume)
        networkToolbar.addAction(volumeAction)        

        #Add Score Progress bar        
        style2 = "QProgressBar::chunk {background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,stop: 0 #F10350,stop: 0.4999 #FF3320,stop: 0.5 #FF0019,stop: 1 #F0F150 );}"
        style3 = "QProgressBar::chunk {background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,stop: 0 #F10350,stop: 0.4999 #FF3320,stop: 0.5 #FF0019,stop: 1 #05F150 );}"
        nonrctpro = QProgressBar()
        nonrctpro.setRange(0, 10000)
        nonrctpro.setValue(10000)        
        rctpro = QProgressBar()
        rctpro.setRange(0, 10000)
        rctpro.setValue(10000)
        self.score_group = QVBoxLayout()           
        self.score_group.addWidget(nonrctpro)
        self.score_group.addWidget(rctpro)
        groupbox_score = QGroupBox()
        groupbox_score.setLayout(self.score_group)
        networkToolbar.addWidget(groupbox_score)

        networkToolbar.addSeparator()
        self.save_screen = QAction(QIcon(icon_path + "/051-printer-1.png"), "Capture", self)        
        self.save_screen.triggered.connect(self.GetScreenShot)
        networkToolbar.addAction(self.save_screen)
        networkToolbar.addSeparator()

        #Predict On, Of
        self.trainAction = QAction(QIcon(icon_path + "/051-pantone-2.png"), "Predict Off", self)
        self.trainAction.setCheckable(True)
        self.trainAction.toggled.connect(self.TogglePrediction)
        networkToolbar.addAction(self.trainAction)

    def InitRendererView(self, layout):

        self.renderViewWidget = E_MainRenderingWidget()
        layout.addWidget(self.renderViewWidget)        


        #Initialize Renderers
        self.m_vtkWidget = QVTKRenderWindowInteractor();
        self.renderViewWidget.AddMainRenderer(self.m_vtkWidget)


        for i in range(3):            
            #Slice View            
            self.m_vtkSliceWidget[i] = QVTKRenderWindowInteractor();

            #Slice Image Slider
            self.m_sliceSlider[i] = QSlider(Qt.Horizontal)
            self.m_sliceSlider[i].setRange(0, 0)
            self.m_sliceSlider[i].setSingleStep(1)

            self.m_sliceSlider[i].rangeChanged.connect(self.onSliderRangeChanged)
            self.m_sliceSlider[i].valueChanged.connect(self.onSliderValueChanged)

            self.renderViewWidget.AddSliceRenderer(self.m_vtkSliceWidget[i])        



    def InitCentralWidget(self):
        MainLayout = QHBoxLayout()
        MainLayout.setSpacing(0)
        MainLayout.setContentsMargins(0,0,0,0)
        self.m_centralWidget.setLayout(MainLayout)


        leftWidget = QWidget()
        leftWidget.setMaximumWidth(350)
        leftLayout = QVBoxLayout()
        leftWidget.setLayout(leftLayout)
        MainLayout.addWidget(leftWidget)



        self.m_treeWidget = E_VolumeTreeWidget(self)
        leftLayout.addWidget(self.m_treeWidget)


        ## ADD Crop widgets
        self.m_croppingWidget = QWidget()        
        cropLayout = QVBoxLayout()
        self.m_croppingWidget.setLayout(cropLayout)

        self.m_cropRenderer = QVTKRenderWindowInteractor();
        cropLayout.addWidget(self.m_cropRenderer)

        self.m_rangeSlider = [0, 0]
        self.m_rangeSlider[0] = QSlider(Qt.Horizontal)
        self.m_rangeSlider[0].setRange(0.0, 1000)
        self.m_rangeSlider[0].setSingleStep(1)
        self.m_rangeSlider[0].setSliderPosition( 500 )
        cropLayout.addWidget(self.m_rangeSlider[0])
        self.m_rangeSlider[0].valueChanged.connect(self.onRangeSliderValueChanged)

        self.m_rangeSlider[1] = QSlider(Qt.Horizontal)
        self.m_rangeSlider[1].setRange(0.0, 1000)
        self.m_rangeSlider[1].setSingleStep(1)
        self.m_rangeSlider[1].setSliderPosition( 500 )
        cropLayout.addWidget(self.m_rangeSlider[1])
        self.m_rangeSlider[1].valueChanged.connect(self.onRangeSliderValueChanged)

        leftLayout.addWidget(self.m_croppingWidget)
        
        


        #Initialize Main View
        self.InitRendererView(MainLayout)                
        


        MainLayout.setStretch(0, 1)
        MainLayout.setStretch(1, 1)
        MainLayout.setStretch(2, 5)        

    def InitManager(self):
        self.Mgr = E_Manager(self)
        self.volumeWidget.SetManager(self.Mgr)


    def onImportVolume(self):
        self.Mgr.SetLog('import Volume')

        #Get Selected Path
        path = QFileDialog.getExistingDirectory(self, "Import 3D Objects", self.m_saveDir)
        
        #Save SaveDir
        self.m_saveDir = path
        with open(os.path.join(root_path, 'res', 'path_tmp'), 'w') as text_file:
            print(self.m_saveDir, file=text_file)
        dirName = str(path).lower()


        try :
            self.Mgr.VolumeMgr.ImportVolume(path)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.Mgr.SetLog(str(e), error=True)

        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

    def TogglePrediction(self, pred):        
        self.Mgr.m_bPred = pred
        
        if pred:
            self.trainAction.setText("Predict On")
        else:
            self.trainAction.setText("Predict Off")        


    def onRandomPred(self):        
        self.Mgr.RandomPrediction()
        self.Mgr.Redraw()

    def onCroppingViewState(self, state):        
        if state == 2: #show
            self.m_croppingWidget.show()
        else:
            self.m_croppingWidget.hide()

    def onVolumeTreeState(self, state):
        if state == 2: #show
            self.m_treeWidget.show()
        else:
            self.m_treeWidget.hide()    
    def onClassActivationMapState(self, state):
        self.Mgr.VolumeMgr.ToggleClassActivationMap(state)


    def onSliderRangeChanged(self, min, max):
        obj = self.sender()
        obj.setSliderPosition( int((min + max) / 2) )

    def onSliderValueChanged(self, value):
        obj = self.sender()
        idx = self.m_sliceSlider.index(obj)

        self.Mgr.VolumeMgr.ChangeSliceIdx(idx, obj.value())


    def onRangeSliderValueChanged(self, value):
        xPos = self.m_rangeSlider[0].value() / 1000
        yPos = self.m_rangeSlider[1].value() / 1000

        self.Mgr.VolumeMgr.UpdateVolumeDataCrop(xPos, yPos)
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

    def keyPressEvent(self, e):

        if e.key() == 65:
            self.m_rangeSlider[0].setValue(self.m_rangeSlider[0].value()-1)
        elif e.key() == 83:
            self.m_rangeSlider[0].setValue(self.m_rangeSlider[0].value()+1)
        elif e.key() == 90:
            self.m_rangeSlider[1].setValue(self.m_rangeSlider[0].value()-1)
        elif e.key() == 88:
            self.m_rangeSlider[1].setValue(self.m_rangeSlider[0].value()+1)
        else:
            return
    


    def SetViewModeNromal(self):
        self.renderViewWidget.SetViewMainView()

    def SetViewModeGrid(self):
        self.renderViewWidget.SetViewGridView()

    def onProgress(self, progress):
        self.progressBar.setValue(progress)

    def onMessage(self, msg):
        self.statusBar().showMessage(msg)

    def UpdateRenderer(self):        
        self.Mgr.RotateCamera()
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()


    def PredictROI(self):
        self.Mgr.PredictROI()


    def onSaveSliceImage(self):
        self.Mgr.SaveSliceImage()

    def SetProgressScore(self, score, label=-1):

        msg = [
            "None RCT " + '{:.2f}'.format(score[0]*100.0) + "%",
            "RCT " + '{:.2f}'.format(score[1]*100.0) + "%"
        ]

        


        if not label == -1:

            pred_class = np.argmax(score)
            if int(label) == pred_class:
                msg[label] = "(correct) " + msg[label]
            else:
                msg[label] = "(wrong) " + msg[label]

            not_idx = not label
            self.score_group.itemAt(label).widget().setStyleSheet("QProgressBar::chunk{ background-color: green; }")
            self.score_group.itemAt(not_idx).widget().setStyleSheet("QProgressBar::chunk{ background-color: red; }")
        else:
            self.score_group.itemAt(0).widget().setStyleSheet("QProgressBar::chunk{ background-color: #1a80d7; }")
            self.score_group.itemAt(1).widget().setStyleSheet("QProgressBar::chunk{ background-color: #1a80d7; }")

        self.score_group.itemAt(0).widget().setValue(score[0] * 10000.0)
        self.score_group.itemAt(0).widget().setFormat(msg[0])
        self.score_group.itemAt(1).widget().setValue(score[1] * 10000.0)
        self.score_group.itemAt(1).widget().setFormat(msg[1])

    def GetScreenShot(self):        
    
        savers = [self.m_vtkWidget.GetRenderWindow(), self.m_vtkSliceWidget[0].GetRenderWindow(), self.m_vtkSliceWidget[1].GetRenderWindow(), self.m_vtkSliceWidget[2].GetRenderWindow()]
        save_name = ["_main.png", "_axl.png", "_cor.png", "_sag.png"]
        original_size = []

        png_writer = vtk.vtkPNGWriter()
        image_filter = vtk.vtkWindowToImageFilter()
        image_filter.SetInputBufferTypeToRGB()


        dir_path = QFileDialog.getSaveFileName(self, "Save Captured Image Directory", self.m_capDir + '/capture', "Image(*.png)")
        if dir_path[0] == "": 
            return
    

        self.m_capDir = os.path.dirname(dir_path[0])
        with open(os.path.join(root_path, 'res', 'capture_path'), 'w') as text_file:
            print(self.m_capDir, file=text_file)



        for idx, ren_win in enumerate(savers):
            original_size.append(ren_win.GetSize())

            image_filter.SetInput(ren_win)
            image_filter.Update()

            png_writer.SetFileName(dir_path[0][:-4]+save_name[idx])
            png_writer.SetInputConnection(image_filter.GetOutputPort())
            png_writer.Write()



