from tkinter import Tk, Label, Button, Frame, StringVar
import distinctipy


class GUI:
    def __init__(self, feature_list, get_label_fun, label_fun_param):
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

        for i in range(rows):
            for j in range(columns):
                index = i*columns + j
                hue = index/(rows*columns)
                rgbval = tuple(int(tup*255) for tup in rgbvals[index])
                hexval = '#%02x%02x%02x' %(rgbval)
                btn = Button(frame, text=feature_list[index], bg=hexval)
                btn.grid(row=i, column=j, sticky="news")

        btn = Button(frame, text="Non-feature")
        btn.grid(row=rows, column=0, columnspan=2, sticky="news")

        btn = Button(frame, text="Close", command=lambda: self.exit(root))
        btn.grid(row=rows+1, column=0, columnspan=2, sticky="news")

        frame.columnconfigure(tuple(range(columns)), weight=1)
        frame.rowconfigure(tuple(range(rows+2)), weight=1)

        #self.master = root
        self.master.after(1000, self.do_something, get_label_fun, label_fun_param)

        root.mainloop()

    def update_label(self, get_label_fun, label_fun_param):
        feat_type = get_label_fun(label_fun_param)
        self.text.set(feat_type)

    def do_something(self, get_label_fun, label_fun_param):
        self.update_label(get_label_fun, label_fun_param)
        self.master.after(1000, self.do_something, get_label_fun, label_fun_param)

    def exit(self, root):
        print("hi")
        root.quit()
