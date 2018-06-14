import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import numpy as np
import random
import scipy.ndimage
from time import gmtime, strftime
from PyQt5.QtWidgets import QApplication
from manager.InteractorStyle import E_InteractorStyle
from manager.InteractorStyle import E_InteractorStyle2D
from manager.InteractorStyle import E_InteractorStyleCropper

from manager.VolumeMgr import E_VolumeManager
from manager.E_SliceRenderer import *
from manager.E_CroppingRenderer import *
import matplotlib.pyplot as plt
from res import labels
import tensorflow as tf
# import network.VRN_64_TF as config_module

#define argument path
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
weight_path = os.path.join(root_path, "res", "cnn", "2block_49")
v_res = 1

class E_Manager:
    def __init__(self, mainFrm):
        self.mainFrm = mainFrm
        self.VolumeMgr = E_VolumeManager(self)
        self.renderer = None
        self.m_sliceRenderer = [0, 0, 0]

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
        self.m_bPred = False


        #Initialize Main Renderer With interactor        
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.0, 0.0, 0.0)
        self.mainFrm.m_vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.mainFrm.m_vtkWidget.GetRenderWindow().Render()
        self.mainFrm.m_vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle( E_InteractorStyle(self))

        #Add Cropping Renderer
        self.cropping_renderer = E_CroppingRenderer(self)
        interactor = E_InteractorStyleCropper(self)
        self.cropping_renderer.SetBackground(0.1, 0.02, 0.2)
        self.mainFrm.m_cropRenderer.GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)
        interactor.AddRenderer(self.cropping_renderer)


        #Add Slice Renderer
        for i in range(3):            
            self.m_sliceRenderer[i] = E_SliceRenderer(self,i)            

        for i in range(3):            
            rendererIdx = (i+1)%3
            interactor = E_InteractorStyle2D(self, rendererIdx)
            self.mainFrm.m_vtkSliceWidget[i].GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)
            interactor.AddRenderer(self.m_sliceRenderer[rendererIdx])
            

        #Initialize
        self.InitObject()        
        self.InitNetwork()


    def InitObject(self):
        #Orientatio WIdget
        axis = vtk.vtkAxesActor()
        self.orWidget = vtk.vtkOrientationMarkerWidget()
        self.orWidget.SetOutlineColor(0.9300, 0.5700, 0.1300)
        self.orWidget.SetOrientationMarker(axis)
        self.orWidget.SetInteractor(  self.mainFrm.m_vtkWidget.GetRenderWindow().GetInteractor() )
        self.orWidget.SetViewport(0.0, 0.0, 0.3, 0.3)
        
        self.Redraw()
        self.Redraw2D()    
 
                
    def VoxelizeObject(self, source):
        #Transform Polydata around Z-axis
        trans = vtk.vtkTransform()
        trans.RotateWXYZ(-90.0, 0, 0, 1.0)
        transFilter = vtk.vtkTransformPolyDataFilter()
        transFilter.SetTransform(trans)
        transFilter.SetInputConnection(source.GetOutputPort())
        transFilter.Update()

        poly = vtk.vtkPolyData()
        poly.DeepCopy(transFilter.GetOutput())

        #Set Voxel Space Resolution nxnxn
        resolution = self.VolumeMgr.resolution
        bounds = [0, 0, 0, 0, 0, 0]
        center = poly.GetCenter()
        poly.GetBounds(bounds)


        #Get Maximum Boundary Length
        maxB = 0.0
        for i in range(0, 6, 2):
            if abs(bounds[i] - bounds[i+1]) > maxB:
                maxB = abs(bounds[i] - bounds[i+1])

        #Calculate Spacing
        spacingVal = maxB / resolution
        spacing = [spacingVal, spacingVal, spacingVal]

        bounds = [center[0] - resolution * spacing[0] / 2, center[0] + resolution * spacing[0] / 2,center[1] - resolution * spacing[1] / 2, center[1] + resolution * spacing[2] / 2, center[2] - resolution * spacing[2] / 2, center[2] + resolution * spacing[0] / 2]

        imgData = vtk.vtkImageData()
        imgData.SetSpacing(spacing)
        origin = [center[0] - resolution * spacing[0] / 2, center[1] - resolution * spacing[1] / 2, center[2] - resolution * spacing[2] / 2]
        imgData.SetOrigin(origin)

        #Dimensions
        dim = [resolution, resolution, resolution]
        imgData.SetDimensions(dim)
        imgData.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)
        imgData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        for i in range(imgData.GetNumberOfPoints()):
            imgData.GetPointData().GetScalars().SetTuple1(i, 1)

        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(poly)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(imgData.GetExtent())

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(imgData)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()


        scalarData = vtk_to_numpy( imgstenc.GetOutput().GetPointData().GetScalars() )
        self.DrawVoxelArray(scalarData)

        self.PredictObject(scalarData)

    def Redraw(self):
        self.mainFrm.m_vtkWidget.GetRenderWindow().Render()
        self.orWidget.SetEnabled(1)        

    def Redraw2D(self):        
        self.cropping_renderer.GetRenderWindow().Render()
        for i in range(3):
            self.m_sliceRenderer[i].GetRenderWindow().Render()
            # self.sliceOrWidget[i].SetEnabled(1)
        

    def InitNetwork(self):        
        
        tf.saved_model.loader.load(self.sess, ['foo-tag'], weight_path)

        self.tensor_in = tf.get_collection('input_tensor')[0]
        y = tf.get_collection('output_tensor')[0]
        self.keep_prob = tf.get_collection('keep_prob')[0]
        # print()

        y = tf.layers.flatten(y)
        self.pred_classes = tf.argmax(y, axis=1)
        self.pred_probs = tf.nn.softmax(y)


        last_conv = tf.get_collection('last_conv')[0][0]
        last_weight = tf.get_default_graph().get_tensor_by_name('fc/kernel:0')[:,:,:,:,1]

        self.class_activation_map =tf.nn.relu( tf.reduce_sum(tf.multiply(last_weight, last_conv), axis=3))


    def predict_tensor(self, input):
        return self.sess.run([self.pred_classes, self.pred_probs, self.class_activation_map], feed_dict={self.tensor_in:input, self.keep_prob:1.0})
        

    def ShowScore(self, label, softmax):        
        self.mainFrm.SetProgressScore(softmax[0], label)

    def PredictObject(self, inputData, label = -1):
        if not self.m_bPred:             
            self.VolumeMgr.RemoveClassActivationMap()
            return

        resolution = self.VolumeMgr.resolution
        inputData = np.asarray(inputData.reshape(1, resolution, resolution, resolution, 1), dtype=np.float32)        

        predict_result = self.predict_tensor(inputData)
        pred_class = predict_result[0]
        
        softmax = predict_result[1]
        self.ShowScore(label, softmax)

        #Class Activation Map         
        activation_map = predict_result[2]
        deconv_rate =  64 / activation_map.shape[0]

        activation_map = scipy.ndimage.zoom(activation_map, deconv_rate)

        activation_map = activation_map / 8
        log = "min : " + str(np.amin(activation_map)) + ", max : " + str(np.amax(activation_map))
        self.SetLog(log)
        activation_map *= 255.0
        activation_map = activation_map.astype(int)
        self.VolumeMgr.AddClassActivationMap(activation_map)


    def MakeDataMatrix(self, x, intensity):
        return intensity*np.repeat(np.repeat(np.repeat(x[0][0], v_res, axis=0), v_res, axis=1), v_res, axis=2)    

    def DrawVoxelArray(self, arrayBuffer):
        #reshape
        resolution = self.VolumeMgr.resolution
        sample = arrayBuffer.reshape(1, 1, resolution, resolution, resolution)
        dataMatrix = self.MakeDataMatrix( np.asarray(sample, dtype=np.uint8), 255)

        data_string = dataMatrix.tostring()
        dataImporter = vtk.vtkImageImport()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, int(dim * v_res)-1, 0, int(dim * v_res)-1, 0, int(dim * v_res)-1)
        dataImporter.SetWholeExtent(0, int(dim * v_res)-1, 0, int(dim * v_res)-1, 0, int(dim * v_res)-1)

        self.VolumeMgr.AddVolumeData(dataImporter.GetOutputPort())
        #Display BoundignBox
        boundingBox = vtk.vtkOutlineFilter()
        boundingBox.SetInputData(dataImporter.GetOutput())

        bbmapper = vtk.vtkPolyDataMapper()
        bbmapper.SetInputConnection(boundingBox.GetOutputPort())

        bbActor = vtk.vtkActor()
        bbActor.SetMapper(bbmapper)
        bbActor.GetProperty().SetColor(1, 0, 0)

        self.renderer.AddActor(bbActor)

        self.Redraw()

    def RunGenerativeMode(self):
        self.SetLog("Generative Mode")
        self.SetLog("Reset Renderer")
        self.SetLog("Set View Mode 1view")
        self.SetLog("Run Generative Mode")

    def ClearScene(self):        
        self.VolumeMgr.RemoveVolumeData()
        self.VolumeMgr.RemoveClassActivationMap()
        
    def RotateCamera(self):
        camera = self.renderer.GetActiveCamera()        
        camera.Azimuth(1)
        camera.SetViewUp(0.0, 1.0, 0.0)

    def SetLog(self, *kwargs):
        log = ""
        for txt in kwargs:
            log += txt + " "        

        QApplication.processEvents() 
        self.mainFrm.statusBar().showMessage(log)

    def PredictROI(self):
        selectedVolume = self.VolumeMgr.m_volumeArray
        shape = selectedVolume.shape
        inputData = np.asarray(selectedVolume.reshape(1, 1, shape[0], shape[1], shape[2]), dtype=np.float32)
        self.SetLog("ROI Prediction")

    def SaveSliceImage(self):
        if self.VolumeMgr.m_resampledVolumeData.any() == None:
            self.SetLog("No Resampled Volume Data is being rendered")
            return

        save_directory = os.path.join(root_path, "humerus_detector", "none_humerus_data")
        data = self.VolumeMgr.m_resampledVolumeData
        
        slice_data = []
        slice_data.append(data[32])
        slice_data.append(np.rot90(data, axes=(0,1))[32])
        slice_data.append(np.rot90(data, axes=(0,2))[32])

        slice_data = np.array(slice_data)
        self.SetLog("slice Data Dim : " + str(slice_data.shape))


        save_directory = os.path.join(root_path, "humerus_detector", "none_humerus_data")
        fname = strftime("%m-%d-%H:%M:%S", gmtime())
        np.savez_compressed(os.path.join(save_directory, fname), features=slice_data)