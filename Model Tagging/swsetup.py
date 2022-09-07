import win32com.client as win32
import re
import pythoncom


def open_file():
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

    return app


def get_selection(app):
    doc = app.ActiveDoc
    selMgr = doc.SelectionManager

    featType = selMgr.GetSelectedObjectType3(1, -1)

    if featType is not None:
        return featType
    else:
        return None
