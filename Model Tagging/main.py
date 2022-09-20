import gui
import swsetup
import swinterface

try:
    import swconst
except ImportError:
    import setup
    setup.run()

featureList = ["Simple Hole", "Closed Pocket", "Countersunk Hole", "Opened Pocket", "Counterbore Hole",
                "Closed Island", "Counterdrilled Hole", "Opened Island", "Tapered Hole", "Inner Fillet", "Closed Slot",
                "Outer Fillet", "Opened Slot", "Inner Chamfer", "Floorless Slot", "Outer Chamfer"]

app = swsetup.start_sw()

files = swsetup.Files(app)
file = files.open_file()
files.open_file()

swi = swinterface.FeatureTag(app, featureList)
my_gui = gui.GUI(featureList, lambda: swsetup.get_selection(app), swi, files)
