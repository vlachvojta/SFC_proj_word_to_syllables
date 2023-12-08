import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def main():
    App()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Hyphenation program")
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+0+0")
        self.config(bg="lightgrey")
        self.notebook = Notebook(self)

        self.frame_gru_complex = FrameComplexGru(self.notebook, width=w, height=h, bg='cyan')
        self.frame_gru_simple = FrameSimpleGru(self.notebook, width=w, height=h, bg='white')
        self.frame_refference = FrameRefference(self.notebook, width=w, height=h, bg='darkgrey')

        self.frame_gru_complex.pack(fill='both')
        self.frame_gru_simple.pack(fill='both')
        self.frame_refference.pack(fill='both')

        self.notebook.add(self.frame_gru_complex, text='Complex GRU')
        self.notebook.add(self.frame_gru_simple, text='Simple GRU')
        self.notebook.add(self.frame_refference, text='Refference')
        
        self.mainloop()


class Notebook(ttk.Notebook):
    def __init__(self, master): # , width, height, bg):
        super().__init__(master)
        self.pack(fill='both')


class FrameComplexGru(tk.Frame):
    def __init__(self, master, width, height, bg):
        super().__init__(master, width=width, height=height, bg=bg)
        # super().__init__(master)
        # self.pack(fill='both')

        label = tk.Label(self, text="PyTorch GRU", font=("Times New Roman", 24))
        label.grid(column=1, row=1, columnspan=4, padx=20, pady=20, sticky='n')

        self.entry = tk.Entry(self, font=("Times New Roman", 16))
        self.entry.grid(column=1, row=2, padx=20, pady=20, sticky='en')
        # self.entry.bind('<Return>', self.btn_transcribe_on_click)

        button = tk.Button(self, text="Transcribe", font=("Times New Roman", 16),
                           command=self.btn_transcribe_on_click)
        button.grid(column=2, row=2, padx=20, pady=20, sticky='wn')

        img_path = 'docs/gru_diagram_adresa.png'
        image = Image.open(img_path)
        photo = ImageTk.PhotoImage(image)

        label = tk.Label(self, image = photo)
        label.image = photo
        label.grid(row=2, column=3, columnspan=2, rowspan=2, padx=20, pady=20, sticky='ne')

    def btn_transcribe_on_click(self):
        word = self.entry.get()
        self.entry.delete(0, tk.END)

        if word:
            print(word)


class FrameSimpleGru(tk.Frame):
    def __init__(self, master, width, height, bg):
        super().__init__(master, width=width, height=height, bg=bg)
        self.pack(fill='both')

        label = tk.Label(self, text="Simple PyTorch GRU", font=("Times New Roman", 24))
        label.pack(padx=20, pady=20)

        self.entry = tk.Entry(self, font=("Times New Roman", 16))
        self.entry.pack(padx=20, pady=20)
        # self.entry.bind('<Return>', self.btn_transcribe_on_click)

        button = tk.Button(self, text="Transcribe", font=("Times New Roman", 16),
                           command=self.btn_transcribe_on_click)
        button.pack(padx=20, pady=20)

    def btn_transcribe_on_click(self):
        word = self.entry.get()
        self.entry.delete(0, tk.END)

        if word:
            print(word)


class FrameRefference(tk.Frame):
    def __init__(self, master, width, height, bg):
        super().__init__(master, width=width, height=height, bg=bg)


if __name__ == '__main__':
    main()