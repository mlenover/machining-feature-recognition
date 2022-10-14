import win32com.client as win32
import pythoncom
import re
import numpy as np
try:
    from swconst import constants
except ImportError:
    import setup
    setup.run()
#from comtypes import automation

#sw_const = win32.gencache.EnsureModule('{4687F359-55D0-4CD3-B6CF-2EB42C11F989}', 0, 29, 0).constants

def open_file():
    global app
    app = win32.Dispatch("SldWorks.Application")
    app.Visible = True
    filter_string = "SOLIDWORKS Part (*.sldprt)|*.sldprt|" \
             "STEP AP203/214/242 (*.step;*.stp)|*.step;*.stp|All Files (*.*)|*.*|"

    open_options = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    config_name = display_name = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_BSTR, "")

    file_name = app.GetOpenFileName("File to Attach", "", filter_string, open_options, config_name, display_name)
    file_error = file_warning = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

    if re.search('(?i:sldprt$)', file_name):
        app.OpenDoc6(file_name, 1, 1, "", file_error, file_warning)

    elif re.search('(?i:(?:step|stp)$)', file_name):
        sw_import_step_data = app.GetImportFileData(file_name)
        sw_import_step_data.MapConfigurationData = True
        app.LoadFile4(file_name, "r", sw_import_step_data, file_error)

    else:
        return None

    model = app.ActiveDoc
    a_bodies = model.GetBodies2(0, True)

    return a_bodies


def assign_face_ids(sw_body):
    id_counter = 0
    sw_face = sw_body.GetFirstFace()
    while True:
        sw_face.SetFaceId(id_counter)
        sw_face = sw_face.GetNextFace

        if sw_face is None:
            break

        id_counter = id_counter + 1


def get_face_type(sw_face):
    face_types = [[sw_const.PLANE_TYPE, "PLANE_TYPE", 6],
                  [sw_const.CYLINDER_TYPE, "CYLINDER_TYPE", 5],
                  [sw_const.CONE_TYPE, "CONE_TYPE", 4],
                  [sw_const.SPHERE_TYPE, "SPHERE_TYPE", 7],
                  [sw_const.TORUS_TYPE, "TORUS_TYPE", 8],
                  [sw_const.BSURF_TYPE, "BSURF_TYPE", 1],
                  [sw_const.BLEND_TYPE, "BLEND_TYPE", 2],
                  [sw_const.OFFSET_TYPE, "OFFSET_TYPE", 3],
                  [sw_const.EXTRU_TYPE, "EXTRU_TYPE", 9],
                  [sw_const.SREV_TYPE, "SREV_TYPE", 10]]

    sw_surf = sw_face.GetSurface

    face_name = "NONE"
    face_id = 0

    for face_type in face_types:
        if sw_surf.Identity == face_type[0]:
            face_name = face_type[1]
            face_id = face_type[2]
            break

    return [face_name, face_id]


def get_face_curvature(sw_face):
    sw_surf = sw_face.GetSurface

    if sw_surf.IsPlane:
        return ["Flat", 0]

    closest_point = sw_surf.GetClosestPointOn(0, 0, 0)
    eval_point = sw_surf.EvaluateAtPoint(closest_point[0], closest_point[1], closest_point[2])
    a = np.array(closest_point[0:3])
    b = np.array(eval_point[6:9])
    sec_point = a + b*0.01

    sec_point = sw_surf.GetClosestPointOn(sec_point[0], sec_point[1], sec_point[2])

    point_a = np.array(closest_point[0:3])
    point_b = np.array(sec_point[0:3])
    n_a = np.array(eval_point[0:3])

    if sw_face.FaceInSurfaceSense == True:
        n_a = n_a * -1

    curv = np.dot((point_b-point_a), n_a)

    if curv < 0:
        return ["Positive", 1]
    else:
        return ["Negative", 2]


def get_outer_loop_edges(sw_face):
    sw_loop = sw_face.GetFirstLoop
    while sw_loop is not None:
        if sw_loop.IsOuter:
            return sw_loop
        else:
            sw_loop = sw_loop.GetNext
    return None


def get_face_width(compare_width, sw_face):
    sw_loop = get_outer_loop_edges(sw_face)
    base_face_ID = sw_face.GetFaceId
    sw_edges = sw_loop.GetEdges

    adjacent_faces = []
    for sw_edge in sw_edges:
        sw_faces = sw_edge.GetTwoAdjacentFaces2

        if sw_faces[0].GetFaceId == base_face_ID:
            adjacent_face = sw_faces[1]
        else:
            adjacent_face = sw_faces[0]

        adjacent_faces.append(adjacent_face)

    #check for parallel faces
    parallel_faces_i = []

    for i, first_adjacent_face in enumerate(adjacent_faces):
        for j, second_adjacent_face in enumerate(adjacent_faces[i+1:]):
            if first_adjacent_face.GetSurface.isPlane and second_adjacent_face.GetSurface.isPlane:
                v1 = first_adjacent_face.Normal
                v2 = second_adjacent_face.Normal

                if np.abs(np.dot(v1, v2)) >= 1:
                    parallel_faces_i.append((i, j+i+1))

    if len(parallel_faces_i) != 0:
        first_dist = True

        for p_faces in parallel_faces_i:
            f0 = adjacent_faces[p_faces[0]]
            f1 = adjacent_faces[p_faces[1]]

            var_pos1 = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_VARIANT, None)
            var_pos2 = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_VARIANT, None)

            dist = app.ActiveDoc.ClosestDistance(f0, f1, var_pos1, var_pos2)
            if first_dist:
                min_dist = dist
                first_dist = False
            elif dist < first_dist:
                min_dist = dist

    else:
        num_adj_faces = len(adjacent_faces)
        if num_adj_faces <= 3:
            return ["Shorter", 2]
        else:
            first_dist = True

            var_pos1 = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_VARIANT, None)
            var_pos2 = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_VARIANT, None)

            for i, f0 in enumerate(adjacent_faces):
                for j, f1 in enumerate(adjacent_faces[i+2:]):
                    if not (i == 0 and j == num_adj_faces - 1):
                        dist = app.ActiveDoc.ClosestDistance(f0, f1, var_pos1, var_pos2)
                        if first_dist:
                            min_dist = dist
                            first_dist = False
                        elif dist < min_dist:
                            min_dist = dist

    if min_dist > compare_width:
        return ["Longer", 1]
    else:
        return ["Shorter", 2]


def get_outer_loop(sw_face):
    sw_loop = sw_face.GetFirstLoop
    while not sw_loop.IsOuter:
        sw_loop = sw_loop.GetNext

    return sw_loop, sw_loop.GetEdgeCount


def get_inner_loops(sw_face):
    sw_loop = sw_face.GetFirstLoop
    loops = []

    while sw_loop is not None:
        if not sw_loop.IsOuter:
            loops.append(sw_loop)
        sw_loop = sw_loop.GetNext

    return loops


def get_curve_mid_u_val(sw_co_edge):
    params = sw_co_edge.GetCurveParams

    if params[6] > params[7]:
        mid_param = (params[6] - params[7]) / 2 + params[7]
    else:
        mid_param = (params[7] - params[6]) / 2 + params[6]

    return mid_param


#next two functions adapted from https://help.solidworks.com/2017/english/api/sldworksapi/select_tangent_faces_example_vb.htm
def get_face_normal_at_co_edge(sw_co_edge):
    mid_param = get_curve_mid_u_val(sw_co_edge)

    point = sw_co_edge.Evaluate(mid_param)
    sw_surface = sw_co_edge.GetLoop.GetFace.GetSurface
    params = sw_surface.EvaluateAtPoint(point[0], point[1], point[2])
    this_normal = params[0:3]

    base_face = sw_co_edge.GetLoop.GetFace

    if base_face.FaceInSurfaceSense:
        this_normal = np.array(this_normal) * -1

    return this_normal


def get_vector_along_face(sw_co_edge):
    mid_param = get_curve_mid_u_val(sw_co_edge)
    sw_face_norm = get_face_normal_at_co_edge(sw_co_edge)

    eval_params = sw_co_edge.Evaluate(mid_param)
    point = eval_params[0:3]
    sw_edge_tangent = eval_params[3:6]

    sw_face_vector = np.cross(sw_face_norm, sw_edge_tangent)
    sw_face_vector = sw_face_vector / np.linalg.norm(sw_face_vector)

    return sw_face_vector


def get_face_angle(sw_co_edge):
    sw_partner_co_edge = sw_co_edge.GetPartner

    sw_face_normal = get_face_normal_at_co_edge(sw_co_edge)

    sw_face_vector = get_vector_along_face(sw_co_edge)
    sw_partner_face_vector = get_vector_along_face(sw_partner_co_edge)

    sw_angle_ref_vec = np.cross(sw_face_vector, sw_face_normal)
    sw_angle_ref_vec = sw_angle_ref_vec / np.linalg.norm(sw_angle_ref_vec)

    #print(sw_face_normal)
    #print(sw_partner_normal)
    #print(sw_angle_ref_vec)

    #angle = np.arctan2(np.dot(np.cross(sw_face_normal, sw_partner_normal), sw_angle_ref_vec), np.dot(sw_face_normal, sw_partner_normal))

    angle = np.arctan2(np.dot(np.cross(sw_face_vector, sw_partner_face_vector), sw_angle_ref_vec),
                       np.dot(sw_face_vector, sw_partner_face_vector))
    angle = (angle + 2 * np.pi) % (2 * np.pi)

    return angle


def get_inner_loop_convexity(sw_inner_loops, sw_outer_loop):
    convexity_feature_vector = np.zeros(2)

    if sw_inner_loops is None:
        return convexity_feature_vector

    for sw_inner_loop in sw_inner_loops:
        sw_inner_co_edge = sw_inner_loop.GetFirstCoEdge

        if sw_inner_co_edge.GetCurve is None:
            continue

        sw_outer_co_edge = sw_outer_loop.GetFirstCoEdge

        angle = get_face_angle(sw_inner_co_edge)

        if sw_inner_co_edge.IsCircle and sw_outer_co_edge.IsCircle:
            sw_inner_center_loc = sw_inner_co_edge.CircleParams[0:3]
            sw_outer_center_loc = sw_outer_co_edge.CircleParams[0:3]

            if sw_inner_center_loc == sw_outer_center_loc:
                if angle > np.pi:
                    convexity_feature_vector[0] = 1

        elif angle <= np.pi:
            convexity_feature_vector[1] = 1

    return convexity_feature_vector


def list_face_convexity(sw_loop):
    convexity_list = np.zeros((5, 11))
    sw_first_co_edge = sw_loop.GetFirstCoEdge
    sw_co_edge = sw_first_co_edge

    while True:
        sw_partner_co_edge = sw_co_edge.GetPartner

        partner_face = sw_partner_co_edge.GetLoop.GetFace
        partner_face_type = get_face_type(partner_face)[1]

        angle = get_face_angle(sw_co_edge)

        if angle == np.pi:
            convexity_list[0][partner_face_type] = convexity_list[0][partner_face_type] + 1

        if angle < np.pi:
            convexity_list[1][partner_face_type] = convexity_list[1][partner_face_type] + 1
            convexity_list[3][partner_face_type] = convexity_list[3][partner_face_type] + 1

            if abs(angle - (np.pi / 2)) < 0.0001:
                convexity_list[4][partner_face_type] = convexity_list[4][partner_face_type] + 1
        else:
            convexity_list[2][partner_face_type] = convexity_list[2][partner_face_type] + 1

        sw_co_edge = sw_co_edge.GetNext

        if sw_co_edge == sw_first_co_edge:
            break

    sum_of_rows = np.sum(convexity_list[0:3])
    convexity_list = convexity_list * 10 / sum_of_rows
    convexity_list = np.nan_to_num(convexity_list)
    convexity_list = np.ceil(convexity_list)

    return convexity_list


def get_faces_with_continuity(sw_loop, is_perp):
    cont_faces = []
    sw_first_co_edge = sw_loop.GetFirstCoEdge
    sw_co_edge = sw_first_co_edge

    while True:
        sw_partner_co_edge = sw_co_edge.GetPartner
        this_normal = get_face_normal_at_co_edge(sw_co_edge)
        partner_normal = get_face_normal_at_co_edge(sw_partner_co_edge)

        if abs(np.dot(this_normal, partner_normal)) < 0.0001 and is_perp:
            co_face = sw_partner_co_edge.GetLoop.GetFace
            cont_faces.append(co_face)
        elif abs(np.dot(this_normal, partner_normal)) >= 1 and not is_perp:
            co_face = sw_partner_co_edge.GetLoop.GetFace
            cont_faces.append(co_face)

        sw_co_edge = sw_co_edge.GetNext

        if sw_co_edge == sw_first_co_edge:
            break

    return cont_faces


aBodies = open_file()

for swBody in aBodies:
    assign_face_ids(swBody)
    swFace = swBody.GetFirstFace()
    featureVector = np.zeros(61)

    while True:
        print("")
        face_type = get_face_type(swFace)
        featureVector[0] = face_type[1]
        print(face_type)

        face_curvature = get_face_curvature(swFace)
        featureVector[1] = face_curvature[1]
        print(face_curvature)

        f_mach_width = 10
        face_width_f_mach = get_face_width(f_mach_width, swFace)
        featureVector[2] = face_width_f_mach[1]
        print(face_width_f_mach)

        e_mach_width = 10
        face_width_e_mach = get_face_width(e_mach_width, swFace)
        featureVector[3] = face_width_e_mach[1]
        print(face_width_e_mach)

        outerLoop, numOuterLoopFaces = get_outer_loop(swFace)

        face_convexity = list_face_convexity(outerLoop)
        featureVector[4:59] = face_convexity.flatten()
        print(face_convexity)

        innerLoops = get_inner_loops(swFace)
        inner_convexity = get_inner_loop_convexity(innerLoops, outerLoop)
        featureVector[59:61] = inner_convexity
        print(inner_convexity)

        print(swFace.GetFeature.Name)

        swFace = swFace.GetNextFace

        if swFace is None:
            break