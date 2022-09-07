import win32com.client as win32
import gui
import swsetup

try:
    import swconst
except ImportError:
    import setup
    setup.run()
    import swconst

swconst = swconst.constants

featureList = ["Simple Hole", "Closed Pocket", "Countersunk Hole", "Opened Pocket", "Counterbore Hole",
                "Closed Island", "Counterdrilled Hole", "Opened Island", "Tapered Hole", "Inner Fillet", "Closed Slot",
                "Outer Fillet", "Opened Slot", "Inner Chamfer", "Floorless Slot", "Outer Chamfer"]


app = swsetup.open_file()
my_gui = gui.GUI(featureList, swsetup.get_selection, app)
