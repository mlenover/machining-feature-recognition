try:
    import swconst
except ImportError:
    import setup
    setup.run()

import swinterface
import tkinter as tk


class SelectionGUI:
    def __init__(self, files):
        root = tk.Tk()
        self.master = root

        root.title("A simple GUI")
        root.geometry("720x550")
        root.attributes("-topmost", True)

        frame = tk.Frame(root)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        frame.grid(row=0, column=0, sticky="news")
        grid = tk.Frame(frame)
        grid.grid(sticky="news", column=1, row=1, columnspan=3)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(1, weight=1)

        btn = tk.Button(frame, text="Accept", bg='#CAD1EE', command=lambda: files.next_file())
        btn.grid(row=0, column=0, columnspan=2, sticky="news")

        btn = tk.Button(frame, text="Reject", command=lambda: files.delete_file())
        btn.grid(row=1, column=0, columnspan=2, sticky="news")

        btn = tk.Button(frame, text="Close", command=lambda: self.exit(root, files))
        btn.grid(row=2, column=0, columnspan=2, sticky="news")

        frame.columnconfigure(tuple(range(1)), weight=1)
        frame.rowconfigure(tuple(range(3)), weight=1)

        root.mainloop()

    @staticmethod
    def exit(root, files):
        files.close_file()
        root.quit()


featureList = ["Simple Hole", "Closed Pocket", "Countersunk Hole", "Opened Pocket", "Counterbore Hole",
                "Closed Island", "Counterdrilled Hole", "Opened Island", "Tapered Hole", "Inner Fillet", "Closed Slot",
                "Outer Fillet", "Opened Slot", "Inner Chamfer", "Floorless Slot", "Outer Chamfer"]

print("Starting Solidworks, please wait a moment")
app = swinterface.start_sw()
print("Solidworks opened, please select a directory")
files = swinterface.Files(app)
file = files.open_file()

my_gui = SelectionGUI(files)
