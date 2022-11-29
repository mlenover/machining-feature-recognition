import win32com.client as win32
import re
import pythoncom
from os import walk, remove, rmdir, path
import gui
from swconst import constants


def get_file_list(open_dir):
    open_dir = open_dir.replace('/', "\\")
    files = []
    for root, d_names, f_names in walk(open_dir):
        for file in f_names:
            if check_file(file):
                f = root + '\\' + file
                files.append(f)

    return files


def get_next_file(file_list):
    if len(file_list) == 0:
        return None
    return file_list.pop(0)


def check_file(file_name):
    if file_name[:2] == "~$":
        return None

    elif re.search('(?i:sldprt$)', file_name):
        return 'sldprt'

    elif re.search('(?i:(?:step|stp)$)', file_name):
        return 'step'

    else:
        return None


def start_sw():
    app = win32.Dispatch("SldWorks.Application")
    app.Visible = True
    return app


def get_selection(app):
    doc = app.ActiveDoc
    selMgr = doc.SelectionManager

    featType = selMgr.GetSelectedObjectType3(1, -1)

    if featType is not None:
        return featType
    else:
        return None


class Files:
    def __init__(self, app):
        open_dir = gui.open_directory()
        self.file_list = get_file_list(open_dir)
        self.file = get_next_file(self.file_list)
        self.app = app

    def open_file(self):
        try:
            file_name = self.file
            file_type = check_file(file_name)
            print(file_name)

            file_error = file_warning = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

            if file_type == 'sldprt':
                self.app.OpenDoc6(file_name, 1, 1, "", file_error, file_warning)
                return True

            elif file_type == 'step':
                sw_import_step_data = self.app.GetImportFileData(file_name)
                sw_import_step_data.MapConfigurationData = True
                self.app.LoadFile4(file_name, "r", sw_import_step_data, file_error)
                return True

            else:
                return False
        except:
            print("HELP")

    def close_file(self):
        pathname = self.file
        filetype = path.splitext(pathname)[1]
        pathname = path.splitext(pathname)[0] + '.sldprt'
        self.app.ActiveDoc.SaveAs3(pathname, 0, 2)
        self.app.CloseAllDocuments(True)
        if filetype != '.sldprt' and filetype != '.SLDPRT':
            remove(self.file)

    def delete_file(self):
        self.app.CloseAllDocuments(True)
        remove(self.file)

        directory = path.dirname(self.file)
        for (root, dirs, f_names) in walk(directory):
            if f_names == []:
                rmdir(root)

        self.file = get_next_file(self.file_list)
        if self.file is not None:
            self.open_file()
        else:
            exit()

    def get_file(self):
        return self.file

    def next_file(self):
        self.close_file()
        self.file = get_next_file(self.file_list)

        if self.file is not None:
            self.open_file()
        else:
            exit()

        return self.file


class FeatureTag:
    def __init__(self, app, features):
        self.app = app
        self.colors = []
        self.colordict = []
        self.features = features[:]
        self.features.insert(0, "No Feature")
        self.currFeature = "No Feature"

    def set_colors(self, colors):
        self.colors = colors[:]
        self.colors.insert(0, (0.792156862745098, 0.8196078431372549, 0.9333333333333333))

        self.colordict = dict(zip(self.features, self.colors))

    def set_feature(self, feature):
        self.currFeature = feature

    def update_label(self):
        if self.colors is None or self.features is None:
            return False

        v_face_prop = self.app.ActiveDoc.MaterialPropertyValues
        rgbcolor = self.colordict[self.currFeature]

        tmp = []
        for c in rgbcolor:
            tmp.append(int(c*255))

        rgbcolor = tuple(tmp)

        color = ''.join('{:02X}'.format(c) for c in rgbcolor[::-1])
        color = int(color, base=16)
        self.app.ActiveDoc.SelectedFaceProperties(color, v_face_prop[3], v_face_prop[4],
                                                  v_face_prop[5], v_face_prop[6],
                                                  v_face_prop[7], v_face_prop[8], False,
                                                  self.currFeature)
        success = self.app.ActiveDoc.SelectionManager.GetSelectedObject6(1, -1).DeSelect
