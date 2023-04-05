from tkinter import messagebox, filedialog, Tk
import swinterface
import win32com.client as win32
import math
import csv
import numpy as np
from setup import constants
import pickle
from featureextraction import get_feature_vector

featureList = ["Simple Hole", "Closed Pocket", "Countersunk Hole", "Opened Pocket", "Counterbore Hole",
                "Closed Island", "Counterdrilled Hole", "Opened Island", "Tapered Hole", "Inner Fillet", "Closed Slot",
                "Outer Fillet", "Opened Slot", "Inner Chamfer", "Floorless Slot", "Outer Chamfer"]
rgbvals = pickle.load(open("rgbvals.pickle", "rb"))

root = Tk()
root.deiconify()
root.lift()
root.focus_force()

default_to_non_feature = False

#Prompt user to select new/existing data file
is_existing_file = messagebox.askquestion("Data File Selection", "Would you like to append data to an existing data file?")

root.destroy()

if is_existing_file == "yes":
    # Existing file: prompt user to select file
    #import csv here
    #save data into data variable
    pass
else:
    csvFileName = filedialog.asksaveasfilename(initialfile='Data File.csv', defaultextension='.csv')
    csvFile = open(csvFileName, 'w', newline='')
    pass
    # New file: prompt user to select directory


#Open Solidworks
app = swinterface.start_sw()

# Open file directory
# Iterate through directory
# Get list of files (including checking file extensions)
files = swinterface.Files(app)

# Create data variable array. List of arrays, one per feature
data = [[], []]

# Create the csv writer
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["Class","Face Type","Curvature of Target Face","Width of taget face (for face-machining)","Width of target face (for edge-machining)","Adjacent unknown convexity  unknown geometry faces in an outer loop","Adjacent unknown convexity bezier surface faces in an outer loop","Adjacent unknown convexity bspline surface faces in an outer loop","Adjacent unknown convexity rectangular trimmed  faces in an outer loop","Adjacent unknown convexity conical faces in an outer loop","Adjacent unknown convexity cylindrical faces in an outer loop","Adjacent unknown convexity planar faces in an outer loop","Adjacent unknown convexity spherical  faces in an outer loop","Adjacent unknown convexity toroidal faces in an outer loop","Adjacent unknown convexity linear extrusion surface faces in an outer loop","Adjacent unknown convexity revolved surface faces in an outer loop","Adjacent concave unknown geometry faces in an outer loop","Adjacent concave bezier surface faces in an outer loop","Adjacent concave bspline surface faces in an outer loop","Adjacent concave rectangular trimmed  faces in an outer loop","Adjacent concave conical faces in an outer loop","Adjacent concave cylindrical faces in an outer loop","Adjacent concave planar faces in an outer loop","Adjacent concave spherical  faces in an outer loop","Adjacent concave toroidal faces in an outer loop","Adjacent concave linear extrusion surface faces in an outer loop","Adjacent concave revolved surface faces in an outer loop","Adjacent convex unknown geometry faces in an outer loop","Adjacent convex bezier surface faces in an outer loop","Adjacent convex bspline surface faces in an outer loop","Adjacent convex rectangular trimmed  faces in an outer loop","Adjacent convex conical faces in an outer loop","Adjacent convex cylindrical faces in an outer loop","Adjacent convex planar faces in an outer loop","Adjacent convex spherical  faces in an outer loop","Adjacent convex toroidal faces in an outer loop","Adjacent convex linear extrusion surface faces in an outer loop","Adjacent convex revolved surface faces in an outer loop","Adjacent non-tangent  unknown geometry faces in an outer loop","Adjacent non-tangent  bezier surface faces in an outer loop","Adjacent non-tangent  bspline surface faces in an outer loop","Adjacent non-tangent  rectangular trimmed  faces in an outer loop","Adjacent non-tangent  conical faces in an outer loop","Adjacent non-tangent  cylindrical faces in an outer loop","Adjacent non-tangent  planar faces in an outer loop","Adjacent non-tangent  spherical  faces in an outer loop","Adjacent non-tangent  toroidal faces in an outer loop","Adjacent non-tangent  linear extrusion surface faces in an outer loop","Adjacent non-tangent  revolved surface faces in an outer loop","Adjacent perpendicular  unknown geometry faces in an outer loop","Adjacent perpendicular  bezier surface faces in an outer loop","Adjacent perpendicular  bspline surface faces in an outer loop","Adjacent perpendicular  rectangular trimmed  faces in an outer loop","Adjacent perpendicular  conical faces in an outer loop","Adjacent perpendicular  cylindrical faces in an outer loop","Adjacent perpendicular  planar faces in an outer loop","Adjacent perpendicular  spherical  faces in an outer loop","Adjacent perpendicular  toroidal faces in an outer loop","Adjacent perpendicular  linear extrusion surface faces in an outer loop","Adjacent perpendicular  revolved surface faces in an outer loop","Contain concave inner loop that is not centered on face?","Contain convex inner loop that is centered on face?"])

#For each file...
#Try to open it
while files.open_file():
    print("New File")
    #app.ActiveDoc.GetFirstModelView.EnableGraphicsUpdate = False
    app.ActiveDoc.FeatureManager.ShowBodies()
    bodies = app.ActiveDoc.GetBodies2(constants.swAllBodies, True)
    selMgr = app.ActiveDoc.SelectionManager
    selData = selMgr.CreateSelectData

    # For each body in file...
    for body in bodies:
        # For each face in body...
        face = body.GetFirstFace()
        while face is not None:
            #app.ActiveDoc.ClearSelection2(True)
            #face.Select4(True, selData)
            # Read face name. Is the name one of the standard tags?
            #rgb
            rgbvals = pickle.load(open("rgbvals.pickle", "rb"))
            nonFeatColor = [0.792, 0.82, 0.933]
            faceProperties = face.GetMaterialPropertyValues2(constants.swAllConfiguration, None)
            isFeat = False
            feat = ""

            if all(f != -1.0 for f in faceProperties[0:3]) or default_to_non_feature:
                isFeat = False
                for featIndex, rgbval in enumerate(rgbvals):
                    if all(abs(faceProperties[j]-rgbval[j]) < 0.004 for j in range(3)):
                        isFeat = True
                        feat = featureList[featIndex]

                if isFeat is False:
                    if all(abs(faceProperties[j]-nonFeatColor[j]) < 0.004 for j in range(3)) or default_to_non_feature:
                        feat = "No Feature"

            if feat:
                print(feat)
                
                if feat == "No Feature":
                    classIndex = 0
                else:
                    classIndex = featureList.index(feat) + 1
                
                featVec = get_feature_vector(face, app)
                csvWriter.writerow(np.insert(featVec, 0, classIndex))
                #csvFile.close()
                
                #pass
            
                #csvFile = open(csvFileName, 'w', newline='')
                
                

            #faceName = face.ModelName

            #if faceName in featureList:
            #    print(faceName)

            # If yes, encode feature
            # Find all instances of that feature in other feature matrices (if row in array)
            # If there are any duplicates, remove all instances of the feature vector
            # Otherwise, append data variable to appropriate feature matrix
            # pass
            face = face.GetNextFace
    
#    csvFile.close()
#    pass
#    csvFile = open(csvFileName, 'w', newline='')
#    csvWriter = csv.writer(csvFile)
    print("Remaining files: ", len(files.file_list))
    files.next_file(doSave = False)

csvFile.close()

#For each body in file...
#For each face in body...
#Read face name. Is the name one of the standard tags?
#If yes, encode feature
#Find all instances of that feature in other feature matrices (if row in array)
#If there are any duplicates, remove all instances of the feature vector
#Otherwise, append data variable to appropriate feature matrix

