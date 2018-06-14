import vtk
import numpy as np
import math

class E_CroppingRenderer(vtk.vtkRenderer):
    def __init__(self, mgr):                                       

        self.centerPos = np.array([0.0, 0.0])
        self.selectedPos = np.array([0.0, 0.0])
        self.bounds = [0.0, 0.0, 0.0]
        self.m_bShowGuide = False

    
        self.Mgr = mgr

        self.SetBackground(0.0, 0.0, 0.0)
        self.GetActiveCamera().SetPosition(0.0, 0.0, 100.0)
        self.GetActiveCamera().ParallelProjectionOn()

        self.Initialize()        
        
    def Initialize(self):
        self.centerLineActor = vtk.vtkActor()
        self.polygonActor = vtk.vtkActor()
        self.selectedPositionActor = vtk.vtkActor()        
        self.selectedPositionActor.GetProperty().SetColor([0.7, 0.7, 0.0])
        
        #Initialize Center selection
        self.selected_position_poly = vtk.vtkPolyData()
        self.selected_position_mapper = vtk.vtkPolyDataMapper()
        self.selected_position_mapper.SetInputData(self.selected_position_poly)
        self.selectedPositionActor.SetMapper(self.selected_position_mapper)

        

    def AddGuide(self, bounds = [0.0, 0.0, 0.0]):                

        self.bounds = bounds

        #ADD Outer Line   
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4)
        
        point0 = np.array([0.0, 0.0, bounds[2]])
        point1 = np.array([bounds[0], 0.0, bounds[2]])
        point2 = np.array([bounds[0], bounds[1], bounds[2]])
        point3 = np.array([0.0, bounds[1], bounds[2]])
        points.SetPoint(0, point0)
        points.SetPoint(1, point1)
        points.SetPoint(2, point2)
        points.SetPoint(3, point3)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(5)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(3)
        lines.InsertCellPoint(0)        

        polygon = vtk.vtkPolyData()
        polygon.SetPoints(points)
        polygon.SetLines(lines)

        polygonMapper = vtk.vtkPolyDataMapper()
        polygonMapper.SetInputData(polygon)
        polygonMapper.Update()        
        self.polygonActor.SetMapper(polygonMapper)        

        
        #Add Center Position
        point_bottom = (point0 + point1) / 2.0
        point_top = (point2 + point3) / 2.0
        point_left = (point0 + point3) / 2.0
        point_right = (point1 + point2) / 2.0

        self.centerPos = np.array([point_top[0], point_left[1]])

        centerLinePoints = vtk.vtkPoints()
        centerLinePoints.SetNumberOfPoints(4)
        centerLinePoints.SetPoint(0, point_bottom)
        centerLinePoints.SetPoint(1, point_top)
        centerLinePoints.SetPoint(2, point_left)
        centerLinePoints.SetPoint(3, point_right)
        
        centerLines = vtk.vtkCellArray()
        centerLines.InsertNextCell(2)
        centerLines.InsertCellPoint(0)
        centerLines.InsertCellPoint(1)        
        centerLines.InsertNextCell(2)
        centerLines.InsertCellPoint(2)
        centerLines.InsertCellPoint(3)

        centerLinePoly = vtk.vtkPolyData()
        centerLinePoly.SetPoints(centerLinePoints)
        centerLinePoly.SetLines(centerLines)

        centerLineMapper = vtk.vtkPolyDataMapper()
        centerLineMapper.SetInputData(centerLinePoly)
        centerLineMapper.Update()
        
        self.centerLineActor.SetMapper(centerLineMapper)        


        self.AddActor(self.polygonActor)        
        self.AddActor(self.centerLineActor)

    def RemoveGuide(self):
        self.RemoveActor(self.polygonActor)
        self.RemoveActor(self.selectedPositionActor)
        self.RemoveActor(self.centerLineActor)
        self.m_bShowGuide = False
        

    def UpdateSelectedPosition(self, position = [0, 0]):

        crop_rate = self.Mgr.VolumeMgr.max_crop_rate
        max_bounds = [self.bounds[0] * crop_rate[0], self.bounds[1] * crop_rate[1]]        
        cube_length = [self.bounds[0]-max_bounds[0], self.bounds[1]-max_bounds[1]]
        self.selectedPos = np.array([position[0] - cube_length[0]/2, position[1] - cube_length[0]/2])
        
        if self.selectedPos[0] > max_bounds[0]:
            self.selectedPos[0] = max_bounds[0]
        elif self.selectedPos[0] <= 0:
            self.selectedPos[0] = 0
        if self.selectedPos[1] > max_bounds[1]:
            self.selectedPos[1] = max_bounds[1]
        elif self.selectedPos[1] <= 0:
            self.selectedPos[1] = 0

        #Calculate Rendering BB length
        cube_length = [self.bounds[0]-max_bounds[0], self.bounds[1]-max_bounds[1]]
        lower_bound = [self.selectedPos[0], self.selectedPos[1]]
        upper_bound = [lower_bound[0] + cube_length[0], lower_bound[1]+cube_length[1]]
        
        #Define BB Points
        point_bottom_left = [lower_bound[0], lower_bound[1], self.bounds[2]]
        point_top_left = [lower_bound[0], upper_bound[1], self.bounds[2]]
        point_bottom_right = [upper_bound[0], lower_bound[1], self.bounds[2]]
        point_top_right= [upper_bound[0], upper_bound[1], self.bounds[2]]

        # print(self.selectedPos, self.bounds)
        #Center Line
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4)
        points.SetPoint(0, point_bottom_left)
        points.SetPoint(1, point_top_left)
        points.SetPoint(2, point_bottom_right)
        points.SetPoint(3, point_top_right)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)        
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(3)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(2)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(3)

        self.selected_position_poly.SetPoints(points)
        self.selected_position_poly.SetLines(lines)
        

        # self.centerLineMapper.Update()
        self.GetRenderWindow().Render()


        if not self.m_bShowGuide:
            self.AddActor(self.selectedPositionActor)
            self.m_bShowGuide = True

    def CalculateDiff(self):
        #Select From Original Image
        crop_rate = self.Mgr.VolumeMgr.max_crop_rate
        max_bounds = [self.bounds[0] * crop_rate[0], self.bounds[1] * crop_rate[1]]
        self.Mgr.mainFrm.m_rangeSlider[0].setValue(self.selectedPos[0]/max_bounds[0] * 1000)
        self.Mgr.mainFrm.m_rangeSlider[1].setValue(self.selectedPos[1]/max_bounds[1] * 1000)

    

    def AddViewProp(self, prop):
        self.RemoveGuide()

        orientation = self.Mgr.VolumeMgr.m_orientation
        if orientation == 'AXL':
            self.polygonActor.GetProperty().SetColor([0.8, 0.0, 0.0])
            self.centerLineActor.GetProperty().SetColor([0.4, 0.0, 0.0])
        elif orientation == 'COR':
            self.polygonActor.GetProperty().SetColor([0.0, 0.8, 0.0])
            self.centerLineActor.GetProperty().SetColor([0.0, 0.4, 0.0])
        elif orientation == 'SAG':
            self.polygonActor.GetProperty().SetColor([0.0, 0.0, 0.8])
            self.centerLineActor.GetProperty().SetColor([0.0, 0.0, 0.4])


        bounds = [prop.GetMaxXBound(), prop.GetMaxYBound(), prop.GetMaxZBound()]
        super(E_CroppingRenderer, self).AddViewProp(prop)                
        self.AddGuide(bounds)
        
        