import win32com.client as win32
import swinterface as swi
import gui
from os import path, remove
import pythoncom
from setup import constants

#prompt user to select directory
app = swi.start_sw()
app.SetUserPreferenceToggle(constants.swStepExportFaceEdgeProps, True)
app.SetUserPreferenceIntegerValue(constants.swStepAP, 214)

#recursively search subdirectories for files ending in *sldprt
files = swi.Files(app)

current_file = files.file

while current_file is not None:
    if swi.check_file(current_file) == 'sldprt':
        #open the file
        file_error = file_warning = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        app.OpenDoc6(current_file, 1, 1, "", file_error, file_warning)
        
        pathname = current_file
        filetype = path.splitext(pathname)[1]
        
        #save the file as .step ap214
        pathname = path.splitext(pathname)[0] + '.STEP'
        app.ActiveDoc.SaveAs3(pathname, 0, 2)
        app.CloseAllDocuments(True)
        remove(current_file)
    
    current_file = swi.get_next_file(files.file_list)


