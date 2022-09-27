#git test 2

try:
    import swconst
except ImportError:
    import setup
    setup.run()

import gui
import swinterface

featureList = ["Simple Hole", "Closed Pocket", "Countersunk Hole", "Opened Pocket", "Counterbore Hole",
                "Closed Island", "Counterdrilled Hole", "Opened Island", "Tapered Hole", "Inner Fillet", "Closed Slot",
                "Outer Fillet", "Opened Slot", "Inner Chamfer", "Floorless Slot", "Outer Chamfer"]

print("Starting Solidworks, please wait a moment")
app = swinterface.start_sw()
print("Solidworks opened, please select a directory")
files = swinterface.Files(app)
file = files.open_file()
files.open_file()

swi = swinterface.FeatureTag(app, featureList)
my_gui = gui.GUI(featureList, lambda: swinterface.get_selection(app), swi, files)
