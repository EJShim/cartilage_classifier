
from vtk.util import numpy_support
import itk
import vtk
import ctypes

import numpy as np
import scipy.ndimage
from collections import Counter

import matplotlib.pyplot as plt


ImageType = itk.Image[itk.F, 3]

class E_VolumeManager:
    def __init__(self, Mgr):
        self.Mgr = Mgr
                
        self.m_volumeInfo = None
        self.m_selectedIdx = None
        self.m_selectedImage = None        


        # self.m_reverseSagittal = False
        # self.m_reverseAxial = False
        self.m_shoulderSide = 'L'        
        self.m_orientation = None ##AXL:1, COR:0, SAG:2

        #Selected Volume CFGS
        self.m_colorFunction = vtk.vtkColorTransferFunction()
        self.m_opacityFunction = vtk.vtkPiecewiseFunction()
        self.m_scalarRange = [0.0, 1.0]
        self.m_volumeProperty = vtk.vtkVolumeProperty()
        self.m_imageProperty = vtk.vtkImageProperty()
        self.m_imageProperty.SetInterpolationTypeToLinear()

        #Volume
        self.volume_data = vtk.vtkImageData()
        self.volume_cropper = vtk.vtkImageData()

        self.m_volumeMapper = vtk.vtkSmartVolumeMapper()
        self.m_volume = vtk.vtkVolume()
        self.m_resliceMapper = [0, 0, 0]
        self.m_resliceActor = [0, 0, 0]

        

        #Color MAp Volume
        self.m_bShowCAM = False
        self.m_bShowVolume = False
        self.m_colorMapMapper = vtk.vtkSmartVolumeMapper()
        self.m_colorMapVolume = vtk.vtkVolume()
        self.m_colorMapResliceMapper = [None, None, None]
        self.m_colorMapResliceActor = [None, None, None]

        self.resolution = 64

        #Crop View
        self.croppingMapper = vtk.vtkImageSliceMapper()
        self.croppingActor = vtk.vtkImageSlice()

        for i in range(3):
            self.m_resliceMapper[i] = vtk.vtkImageSliceMapper()
            self.m_resliceActor[i] = vtk.vtkImageSlice()
            self.m_colorMapResliceMapper[i] = vtk.vtkImageSliceMapper()
            self.m_colorMapResliceActor[i] = vtk.vtkImageSlice()        
        
        self.m_resliceActor[0].RotateY(90)
        self.m_resliceActor[1].RotateX(-90)
        self.m_colorMapResliceActor[0] .RotateY(90) 
        self.m_colorMapResliceActor[1] .RotateX(-90)

        #Initialize
        self.SetPresetFunctions(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex())        
        self.InitializeVolumeFunctions()
        self.InitializeClassActivationMap()
        # self.InitializeRenderFunctions()

    def InitializeVolumeFunctions(self):
        #Init Mapper
        self.m_volumeMapper.SetInputData(self.volume_data)        
        self.croppingMapper.SetInputData(self.volume_cropper)
        for i in range(3):
            self.m_resliceMapper[i].SetInputData(self.volume_data)

        # Prepare volume properties.
        self.m_volumeProperty.SetColor(self.m_colorFunction)
        self.m_volumeProperty.SetScalarOpacity(self.m_opacityFunction)

        #Init Actor
        self.m_volume.SetMapper(self.m_volumeMapper)
        self.m_volume.SetProperty(self.m_volumeProperty)
        self.m_volume.SetPosition([0, 0, 0])

        #Init Slice
        for i in range(3):
            self.m_resliceMapper[i].SetOrientation(i)
            self.m_resliceActor[i].SetMapper(self.m_resliceMapper[i])
            self.m_resliceActor[i].SetProperty(self.m_imageProperty)
        
        self.croppingActor.SetMapper(self.croppingMapper)
        self.croppingActor.SetProperty(self.m_imageProperty)

    def SetPresetFunctions(self, idx, update = False):

        #Housefield unit : -1024 ~ 3072
        if update == False:
            self.m_colorFunction.RemoveAllPoints()
            self.m_opacityFunction.RemoveAllPoints()

        housefiledRange = 3072 + 1024
        sRange = self.m_scalarRange[1] - self.m_scalarRange[0]

        self.m_imageProperty.SetColorLevel((self.m_scalarRange[1] + self.m_scalarRange[0])/2)
        self.m_imageProperty.SetColorWindow(self.m_scalarRange[1] - self.m_scalarRange[0]-1)


        rangeFactor = sRange / housefiledRange

        if idx == 0: #MIP
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 1.0, 1.0, 1.0)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 1.0, 1.0, 1.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0], 0.0)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 1.0)

            self.m_volumeProperty.ShadeOff()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToMaximumIntensity()

        elif idx == 1: #CT_SKIN
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 0.0, 0.0, 0.0, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/3.0 , 0.62, 0.36, 0.18, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/2.0 , 0.88, 0.60, 0.29, 0.33, 0.45)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 0.83, 0.66, 1.0, 0.5, 0.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0],0.0, 0.5, 0.0)
            self.m_opacityFunction.AddPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/3.0, 0.0, 0.5, 0.0)
            self.m_opacityFunction.AddPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/2.0, 1.0, 0.33, 0.45)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 1.0, 0.5, 0.0)

            self.m_volumeProperty.ShadeOn()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToComposite()


        elif idx == 2: #CT_BONE
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 0.0, 0.0, 0.0, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/3.0 , 0.73, 0.25, 0.30, 0.49, 0.0)
            self.m_colorFunction.AddRGBPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/2.0 , 0.90, 0.82, 0.56, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 1.0, 1.0, 1.0, 0.5, 0.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0],0.0, 0.5, 0.0)
            self.m_opacityFunction.AddPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/3.0, 0.0, 0.49, 0.61)
            self.m_opacityFunction.AddPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/2.0, 0.72, 0.5, 0.0)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 0.71, 0.5, 0.0)

            self.m_volumeProperty.ShadeOn()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToComposite()

        elif idx == 3: #Voxel
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 0.0, 0.0, 1.0)
            self.m_colorFunction.AddRGBPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/2.0, 0.0, 1.0, 0.0)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 1.0, 0.0, 0.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0], 0.0)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 1.0)

            self.m_volumeProperty.ShadeOn()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToComposite()

    def ImportVolume(self, fileSeries):
        #Series = 0x0020, 0x0011
        #Instance = 0x0020, 0x0013

        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.AddSeriesRestriction("0008|0021")
        namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(fileSeries)
        seriesUID = namesGenerator.GetSeriesUIDs()

        volumedata = []
        metadata = []
    
        for seriesIdentifier in seriesUID:
            fileNames = namesGenerator.GetFileNames(seriesIdentifier)
            reader = itk.ImageSeriesReader[ImageType].New()
            dicomIO = itk.GDCMImageIO.New()
            reader.SetImageIO(dicomIO)
            reader.SetFileNames(fileNames)
            reader.Update()

            ##Make Dictionary
            volumedata.append(reader)
            metadata.append(dicomIO.GetMetaDataDictionary())
                
        patient = dict(volumedata = volumedata, metadata = metadata)
        self.m_volumeInfo = patient
        self.UpdateVolumeTree()
        self.Mgr.ClearScene()
        return

    def ToggleClassActivationMap(self, state):        
        if state == 2:
            self.ShowClassActivationMap()            
        else:            
            self.RemoveClassActivationMap()

        self.Mgr.Redraw()
        self.Mgr.Redraw2D()
    
    def ShowClassActivationMap(self):
        
        if self.m_bShowCAM: return
        #Add To Renderer
        for i in range(3):
            rendererIdx = i
            self.Mgr.m_sliceRenderer[i].AddViewProp(self.m_colorMapResliceActor[i])
            self.Mgr.m_sliceRenderer[i].ResetCamera()
            self.Mgr.m_sliceRenderer[i].GetActiveCamera().Zoom(1.5)

        #Add Actor
        self.Mgr.renderer.AddVolume(self.m_colorMapVolume)
        self.Mgr.renderer.ResetCamera()
        self.m_bShowCAM = True
        
    def RemoveClassActivationMap(self):
        if not self.m_bShowCAM: return
        #Remove From Renderer
        for i in range(3):                
            self.Mgr.m_sliceRenderer[i].RemoveViewProp(self.m_colorMapResliceActor[i])

        #Add Actor
        self.Mgr.renderer.RemoveVolume(self.m_colorMapVolume)
        self.m_bShowCAM = False

    def InitializeClassActivationMap(self):        
        self.cam_data = vtk.vtkImageData()
        self.cam_data.SetOrigin([0, 0, 0])
        self.cam_data.SetDimensions(64,64,64,)
        self.cam_data.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1);
        self.cam_data.SetSpacing([1.0, 1.0, 1.0])

        #set Class Activation Map
        cam_color_function = vtk.vtkColorTransferFunction()
        cam_opacity_function = vtk.vtkPiecewiseFunction()
        scalarRange = [0.0, 255.0]        
        cam_volume_property = vtk.vtkVolumeProperty()

        cam_color_function.AddRGBPoint((scalarRange[0] + scalarRange[1])*0.4, 0.0, 0.0, 1.0)
        cam_color_function.AddRGBPoint((scalarRange[0] + scalarRange[1])*0.7, 0.0, 1.0, 0.0)
        cam_color_function.AddRGBPoint(scalarRange[1], 1.0, 0.0, 0.0)

        cam_opacity_function.AddPoint((scalarRange[0] + scalarRange[1])*0.0, 0.3)
        cam_opacity_function.AddPoint(scalarRange[1], 0.3)

        cam_volume_property.SetColor(cam_color_function)
        cam_volume_property.SetScalarOpacity(cam_opacity_function)
        cam_volume_property.ShadeOff()
        cam_volume_property.SetInterpolationTypeToLinear()

        self.m_colorMapMapper.SetInputData(self.cam_data)
        self.m_colorMapMapper.SetBlendModeToMaximumIntensity()

        #Actor        
        self.m_colorMapVolume.SetMapper(self.m_colorMapMapper)
        self.m_colorMapVolume.SetProperty(cam_volume_property)
        self.m_colorMapVolume.SetPosition([0, 0, 0])

        lookupTable = vtk.vtkLookupTable()
        lookupTable.SetTableRange(0.0, 255.0)
        lookupTable.SetHueRange(0.7, 0.0)
        lookupTable.Build()

        imageProperty = vtk.vtkImageProperty()
        imageProperty.SetInterpolationTypeToLinear()
        imageProperty.SetLookupTable(lookupTable)
        imageProperty.SetOpacity(0.3)

        #Slice
        for i in range(3):
            self.m_colorMapResliceMapper[i].SetInputData(self.cam_data)
            self.m_colorMapResliceMapper[i].SetOrientation(i)
            self.m_colorMapResliceActor[i].SetMapper(self.m_colorMapResliceMapper[i])
            self.m_colorMapResliceActor[i].SetProperty(imageProperty)
            
            
        
    def UpdateClassActivationMap(self, camArray):
        #This Function
        self.cam_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(num_array=camArray.ravel(), deep=True, array_type = vtk.VTK_FLOAT))
        

    def AddClassActivationMap(self, camArray):        
        #This Function
        self.cam_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(num_array=camArray.ravel(), deep=True, array_type = vtk.VTK_FLOAT))

        if not self.m_bShowCAM: 
            self.ShowClassActivationMap()

        self.Mgr.mainFrm.classCheck.setEnabled(True)

    def AddCropSlice(self):
        #Should Be Called After "Add Volume"
        self.volume_cropper.DeepCopy(self.volume_data)
        self.croppingMapper.SetSliceNumber(self.croppingMapper.GetSliceNumberMaxValue()//2)

    def ShowCropSlice(self):
        #Add Cropping        
        self.Mgr.cropping_renderer.AddViewProp(self.croppingActor)
        self.Mgr.cropping_renderer.ResetCamera()
        self.Mgr.cropping_renderer.GetActiveCamera().Zoom(1.5)


    def AddVolume(self, volumeArray, spacing=[1.0, 1.0, 1.0]):
        floatArray = numpy_support.numpy_to_vtk(num_array=volumeArray.ravel(), deep=True, array_type = vtk.VTK_FLOAT)
        
        dim = volumeArray.shape

        #self.volume_data.AllocateScalars(vtk.VTK_UNSIGNVTK_ED_INT, 1);
        self.volume_data.SetOrigin([0,0,0])
        self.volume_data.SetDimensions(dim[2], dim[1], dim[0])        
        self.volume_data.SetSpacing(spacing)
        self.volume_data.GetPointData().SetScalars(floatArray)

        #Get Scalar Range
        self.m_scalarRange = self.volume_data.GetScalarRange()

        #Update Slider
        for i in range(3):
            minVal = self.m_resliceMapper[i].GetSliceNumberMinValue()
            maxVal = self.m_resliceMapper[i].GetSliceNumberMaxValue()
            self.Mgr.mainFrm.m_sliceSlider[i].setRange(minVal, maxVal)   

       

    def ShowVolume(self):        
        #Add Slice
        for i in range(3):            
            #Add SLice
            rendererIdx = i            
            self.Mgr.m_sliceRenderer[rendererIdx].AddViewProp(self.m_resliceActor[i])
            self.Mgr.m_sliceRenderer[rendererIdx].ResetCamera()
            self.Mgr.m_sliceRenderer[rendererIdx].GetActiveCamera().Zoom(1.5)        
            
        #Add Actor
        self.Mgr.renderer.AddVolume(self.m_volume)
        self.Mgr.renderer.ResetCamera()

        self.m_bShowVolume = True

        #Set preset
        self.Mgr.mainFrm.volumeWidget.onChangeIndex(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex(), Update=False)

    def RemoveVolumeData(self):
        if not self.m_bShowVolume: return
        #Add Slice
        for i in range(3):            
            #Add SLice
            rendererIdx = i
            self.Mgr.m_sliceRenderer[rendererIdx].RemoveViewProp(self.m_resliceActor[i])
        #Add Actor
        self.Mgr.cropping_renderer.RemoveViewProp(self.croppingActor)
        self.Mgr.renderer.RemoveVolume(self.m_volume)        
        self.m_bShowVolume = False
        

    def ForwardSliceImage(self, idx):
        sliceNum = self.m_resliceMapper[idx].GetSliceNumber()
        if sliceNum >= self.m_resliceMapper[idx].GetSliceNumberMaxValue():
            return

        # self.ChangeSliceIdx(idx, sliceNum + 1)
        self.Mgr.mainFrm.m_sliceSlider[idx].setValue(sliceNum + 1)

    def BackwardSliceImage(self, idx):
        sliceNum = self.m_resliceMapper[idx].GetSliceNumber()

        sliceNum = self.m_resliceMapper[idx].GetSliceNumber()
        if sliceNum <= self.m_resliceMapper[idx].GetSliceNumberMinValue():
            return

        # self.ChangeSliceIdx(idx, sliceNum - 1)

        #Set Slider Value
        self.Mgr.mainFrm.m_sliceSlider[idx].setValue(sliceNum - 1)


    def ChangeSliceIdx(self, idx, sliceNum):
        self.m_resliceMapper[idx].SetSliceNumber(sliceNum)
        self.m_colorMapResliceMapper[idx].SetSliceNumber(sliceNum)        
        self.Mgr.Redraw2D()

    def UpdateVolumeDataCrop(self, xP, yP):
        #Get Selected ITK image
        selected_image = self.m_selectedImage
        crop_position = [0, 0, 0]

        if self.m_orientation == 'AXL':
            crop_position[0] = int(self.crop_position[0] * xP)
            crop_position[2] = int(self.crop_position[2] * yP)            
        elif self.m_orientation == 'COR':
            crop_position[0] = int(self.crop_position[0] * xP)
            crop_position[1] = int(self.crop_position[1] * yP)
        else:
            crop_position[2] = int(self.crop_position[2] * xP)
            crop_position[1] = int(self.crop_position[1] * yP)        


        #Crop + Resample
        resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
        resampler.SetInput(selected_image)        
        resampler.SetOutputStartIndex(crop_position)
        resampler.SetSize([64,64,64])
        resampler.SetOutputSpacing(self.resample_spacing)
        resampler.SetOutputOrigin(selected_image.GetOrigin())
        resampler.SetOutputDirection(selected_image.GetDirection())
        resampler.UpdateLargestPossibleRegion()
        resampler.Update()
        output_image = resampler.GetOutput()

        #Make Array
        volumeBuffer = itk.GetArrayFromImage(output_image)

        #Add Volume
        self.AddVolume(volumeBuffer, [1, 1, 1])
        self.ShowVolume()

        #Predict if it is enabled
        self.Mgr.PredictObject(volumeBuffer)

        #Redraw
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

        return

    def UpdateVolumeTree(self):
        if self.m_volumeInfo == None: return
        self.Mgr.mainFrm.m_treeWidget.updateTree(self.m_volumeInfo)

    def AddSelectedVolume(self, idx):
        #Clear Scene
        self.Mgr.ClearScene()        
        
        self.m_selectedIdx = idx
        volumedata = self.m_volumeInfo['volumedata'][idx]
        metadata = self.m_volumeInfo['metadata'][idx]
        
        ########################TEMP DICOM TAG INFO READRE############################
        side = 'right'
        imagePositionPatient = metadata["0020|0032"].split('\\')
        if float(imagePositionPatient[0]) > -10.0:
            side='left'

        print(imagePositionPatient, side)
        ########################TEMP DICOM TAG INFO READRE############################

        #Adjust Orientation        
        orienter = itk.OrientImageFilter[ImageType, ImageType].New()
        orienter.UseImageDirectionOn()
        orienter.SetInput(volumedata.GetOutput())
        orienter.Update()
        image = orienter.GetOutput()


        if side == 'left':   
            flipper = itk.FlipImageFilter[ImageType].New()
            flipper.SetInput(image)
            flipper.SetFlipAxes((True, False, False))
            flipper.Update()
            image = flipper.GetOutput()


    
        #Rescale Image Intensity 0 ~ 255
        normalizer = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
        normalizer.SetInput(image)
        normalizer.SetOutputMinimum(0)
        normalizer.SetOutputMaximum(255)
        normalizer.Update()
        itk_image = normalizer.GetOutput()

        self.m_selectedImage = itk_image
        
        volumeBuffer = itk.GetArrayFromImage(itk_image)        
        
        #Adjust Shoulder Side
        # if self.m_shoulderSide == 'L':
        #     volumeArray = np.flip(volumeArray, 2)
        #1,2 = z axes
        #0,2 = y axes
        #0,1 = x axes
        # volumeArray, renderSpacing = self.ResampleVolumeData(volumeArray, spacing)        


        #Update Resample Information        
        size = itk_image.GetLargestPossibleRegion().GetSize()
        spacing = itk_image.GetSpacing()        
        
        #Get Volume Thickness
        length_of_cube = np.amax(spacing) * np.amin(size)
        new_spacing = length_of_cube / 64

        #New Size and Spacing        
        self.resample_spacing = itk.Vector[itk.D,3]([new_spacing, new_spacing, new_spacing])
        
        #Calculate Crop Position && Save Orientation                 
        self.crop_position =  [0,0,0]
        self.max_crop_rate = [0, 0]
        orientation = np.argmin(volumeBuffer.shape)

        #Reset Cropping Actor
        if self.m_orientation == 'AXL':
            self.croppingActor.RotateX(90)            
        elif self.m_orientation == 'SAG':
            self.croppingActor.RotateY(-90)        


        if orientation == 1:
            self.m_orientation = 'AXL'
            self.crop_position[0] = (size[0]*spacing[0]-length_of_cube)/new_spacing
            self.crop_position[2] = (size[2]*spacing[2]-length_of_cube)/new_spacing

            self.max_crop_rate[0] = 1 - length_of_cube/(size[0]*spacing[0])
            self.max_crop_rate[1] = 1 - length_of_cube/(size[2]*spacing[2])

            #Update Cropping Reference
            self.croppingMapper.SetOrientation(1)
            self.croppingActor.RotateX(-90)
                        
        elif orientation == 0:
            self.m_orientation = 'COR'
            self.crop_position[0] = (size[0]*spacing[0]-length_of_cube)/new_spacing
            self.crop_position[1] = (size[1]*spacing[1]-length_of_cube)/new_spacing

            self.max_crop_rate[0] = 1 - length_of_cube/(size[0]*spacing[0])
            self.max_crop_rate[1] = 1 - length_of_cube/(size[1]*spacing[1])

            self.croppingMapper.SetOrientation(2)
            
        else:
            self.m_orientation = 'SAG'
            self.crop_position[2] = (size[2]*spacing[2]-length_of_cube)/new_spacing
            self.crop_position[1] = (size[1]*spacing[1]-length_of_cube)/new_spacing

            self.max_crop_rate[0] = 1 - length_of_cube/(size[2]*spacing[2])
            self.max_crop_rate[1] = 1 - length_of_cube/(size[1]*spacing[1])

            self.croppingMapper.SetOrientation(0)
            self.croppingActor.RotateY(90)

        self.Mgr.SetLog("Reslice Orientaiton : ", str(self.m_orientation))

        #Render Volume
        self.AddVolume(volumeBuffer, spacing=itk_image.GetSpacing())
        self.ShowVolume()

        #Render Crop Slice
        self.AddCropSlice()
        self.ShowCropSlice()
        
        #Redraw
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()
        
        return
