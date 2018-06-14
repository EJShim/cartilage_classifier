
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# import itk

class E_VolumeTreeWidget(QTreeWidget):
    def __init__(self, parent = None):
        super(QTreeWidget, self).__init__(parent)

        self.mainFrm = parent

        self.setSortingEnabled(True)
        self.setHeaderHidden(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setSortingEnabled(False)

        self.itemDoubleClicked.connect(self.dbcEvent)

    def updateTree(self, info):
    
        self.clear()
        parent = QTreeWidgetItem(self)
        parent.setText(0, info['name'])
        self.addTopLevelItem(parent)


        serieses = info['serieses']
        for series in serieses:
            dicomIO = series.GetImageIO()
            metadict = dicomIO.GetMetaDataDictionary()
            series_description = metadict["0008|103e"]

            child = QTreeWidgetItem()
            child.setText(0, series_description)
            parent.addChild(child)

            description = series_description.lower()
            #Fat Supression
            if not description.find('fs/') == -1 or not description.find('fs ') == -1 or not description.find('fat') == -1 or not description.find('f/s') == -1 or description.endswith('fs') or not description.find('fs_') == -1 or not description.find('spir') == -1 or not description.find('spair') == -1:                
                child.setBackground(0, QBrush(QColor('green')))
                
                if not description.find('cor') == -1:
                    child.setForeground(0, QBrush(QColor('red')))
                                        

            elif description.find('t1') == -1 and description.find('t2')== -1:
                child.setBackground(0, QBrush(QColor('red')))
        self.expandAll()


    def dbcEvent(self, item, col):
        #Double-click Event
        if item.parent() == None:
            return
        
        idx = item.parent().indexOfChild(item)        
        self.mainFrm.Mgr.VolumeMgr.AddSelectedVolume(idx)