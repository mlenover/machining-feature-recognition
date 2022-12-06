from setup import constants
import win32com.client as win32
import pythoncom
import swinterface
import featureextraction


app = swinterface.start_sw()
swi = swinterface.Files(app)

swi.open_file()
bodies = app.ActiveDoc.GetBodies2(constants.swAllBodies, True)

def get_plane_properties(sw_surface):
    pass

def get_cylinder_properties(sw_surface):
    pass

def get_cone_properties(sw_surface):
    pass

def get_sphere_properties(sw_surface):
    pass

def get_torus_properties(sw_surface):
    pass

def get_bsurf_properties(sw_surface):
    pass

def get_blend_properties(sw_surface):
    pass

def get_offset_properties(sw_surface):
    pass

def get_torus_properties(sw_surface):
    pass

def get_bsurf_properties(sw_surface):
    pass

faceFunction = {constants.PLANE_TYPE : get_plane_properties,
                constants.CYLINDER_TYPE : get_cylinder_properties,
                constants.CONE_TYPE : get_cone_properties,
                constants.SPHERE_TYPE : get_sphere_properties,
                constants.TORUS_TYPE : get_torus_properties,
                constants.BSURF_TYPE : get_bsurf_properties,
                constants.BLEND_TYPE : get_blend_properties,
                constants.OFFSET_TYPE : get_offset_properties,
                constants.EXTRU_TYPE, "EXTRU_TYPE", 9],
                constants.SREV_TYPE, "SREV_TYPE", 10]]
}

for body in bodies:
    faces = body.GetFaces()
    for face in faces:
        surface = face.GetSurface
        param_obj = surface.Parameterization2
        
        params = []
        
        params.append(param_obj.UMax)
        params.append(param_obj.UMaxBoundType)
        params.append(param_obj.UMin)
        params.append(param_obj.UMinBoundType)
        params.append(param_obj.UProperties)
        params.append(param_obj.UPropertyNumber)
        params.append(param_obj.VMax)
        params.append(param_obj.VMaxBoundType)
        params.append(param_obj.VMin)
        params.append(param_obj.VMinBoundType)
        params.append(param_obj.VProperties)
        params.append(param_obj.VPropertyNumber)
        print(params)
        
        faceType = featureextraction.get_face_type(face)[0]
        faceFunction[faceType]
                