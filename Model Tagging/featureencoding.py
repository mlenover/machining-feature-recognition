from tkinter import messagebox, filedialog
import swinterface
import win32com.client as win32
import math

try:
    from swconst import constants
except ImportError:
    import setup
    setup.run()
    from swconst import constants

import pickle

featureList = ["Simple Hole", "Closed Pocket", "Countersunk Hole", "Opened Pocket", "Counterbore Hole",
                "Closed Island", "Counterdrilled Hole", "Opened Island", "Tapered Hole", "Inner Fillet", "Closed Slot",
                "Outer Fillet", "Opened Slot", "Inner Chamfer", "Floorless Slot", "Outer Chamfer"]
rgbvals = pickle.load(open("rgbvals.pickle", "rb"))

#Prompt user to select new/existing data file
is_existing_file = messagebox.askquestion("Data File Selection", "Would you like to append data to an existing data file?")

if is_existing_file == "yes":
    # Existing file: prompt user to select file
    #import csv here
    #save data into data variable
    pass
else:
    # New file: prompt user to select directory
    folder = filedialog.askdirectory()

#Open Solidworks
app = swinterface.start_sw()

# Open file directory
# Iterate through directory
# Get list of files (including checking file extensions)
files = swinterface.Files(app)

# Create data variable array. List of arrays, one per feature
data = [[], []]

#For each file...
#Try to open it
while files.open_file():
    print("New File")
    app.ActiveDoc.FeatureManager.ShowBodies()
    bodies = app.ActiveDoc.GetBodies2(constants.swAllBodies, True)
    selMgr = app.ActiveDoc.SelectionManager
    selData = selMgr.CreateSelectData

    # For each body in file...
    for body in bodies:
        # For each face in body...
        face = body.GetFirstFace()
        while face is not None:
            app.ActiveDoc.ClearSelection2(True)
            #face.Select4(True, selData)
            # Read face name. Is the name one of the standard tags?
            #rgb
            rgbvals = pickle.load(open("rgbvals.pickle", "rb"))
            faceProperties = face.GetMaterialPropertyValues2(constants.swAllConfiguration, None)
            isFeat = False

            if all(f != -1.0 for f in faceProperties[0:3]):
                isFeat = False
                for featIndex, rgbval in enumerate(rgbvals):
                    if all(abs(faceProperties[j]-rgbval[j]) < 0.004 for j in range(3)):
                        isFeat = True
                        print(featureList[featIndex])

                if isFeat is False:
                    face.Select4(True, selData)
                    print("Non-Feature")


            #faceName = face.ModelName

            #if faceName in featureList:
            #    print(faceName)

            # If yes, encode feature
            # Find all instances of that feature in other feature matrices (if row in array)
            # If there are any duplicates, remove all instances of the feature vector
            # Otherwise, append data variable to appropriate feature matrix
            pass
            face = face.GetNextFace

    files.next_file()

#For each body in file...
#For each face in body...
#Read face name. Is the name one of the standard tags?
#If yes, encode feature
#Find all instances of that feature in other feature matrices (if row in array)
#If there are any duplicates, remove all instances of the feature vector
#Otherwise, append data variable to appropriate feature matrix

