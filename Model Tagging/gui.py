from tkinter import Tk, Label, Button, Frame, StringVar
import distinctipy


class GUI:
    def __init__(self, feature_list, get_label_fun, swi):
        root = Tk()

        self.master = root
        self.text = StringVar()
        self.text.set("No Face Selected")

        root.title("A simple GUI")
        root.geometry("720x550")
        root.attributes("-topmost", True)

        frame = Frame(root)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        frame.grid(row=0, column=0, sticky="news")
        grid = Frame(frame)
        grid.grid(sticky="news", column=1, row=1, columnspan=3)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(1, weight=1)

        columns = 2
        rows = int(len(feature_list) / 2)
        rgbvals = distinctipy.get_colors(rows * columns)
        swi.set_colors(rgbvals)

        for i in range(rows):
            for j in range(columns):
                index = i*columns + j
                rgbval = tuple(int(tup*255) for tup in rgbvals[index])
                hexval = '#%02x%02x%02x' %(rgbval)
                feature = feature_list[index]
                btn = Button(frame, text=feature, bg=hexval, command=lambda f=feature: swi.set_feature(f))
                btn.grid(row=i, column=j, sticky="news")

        btn = Button(frame, text="Non-feature", command=lambda: swi.set_feature("No Feature"))
        btn.grid(row=rows, column=0, columnspan=2, sticky="news")

        btn = Button(frame, text="Close", command=lambda: self.exit(root))
        btn.grid(row=rows+1, column=0, columnspan=2, sticky="news")

        label = Label(root, textvariable=self.text)
        label.grid(row=rows+2, column=0, columnspan=2, padx=5, pady=5)

        frame.columnconfigure(tuple(range(columns)), weight=1)
        frame.rowconfigure(tuple(range(rows+2)), weight=1)

        self.master.after(1000, self.do_something, get_label_fun, swi)

        root.mainloop()

    def update_label(self, get_label_fun):
        feat_type = get_label_fun
        label_text = str(feat_type)
        self.text.set(label_text)

    def do_something(self, get_label_fun, swi):
        selection = get_label_fun()
        if selection == 2:
            swi.update_label()
        self.update_label(selection)
        self.master.after(10, self.do_something, get_label_fun, swi)

    def exit(self, root):
        root.quit()
