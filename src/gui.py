import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from inference import InferenceEngine


def main():
    App(InferenceEngine())


class App(tk.Tk):
    def __init__(self, inference_engine: InferenceEngine):
        super().__init__()
        self.inference_engine = inference_engine

        self.title("Hyphenation program")
        screenwidth, screenheight = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{screenwidth}x{screenheight}+0+0")
        self.config(bg="lightgrey")
        

        self.entry = tk.Entry(self, font=("Times New Roman", 16))
        self.entry.grid(column=3, row=2, padx=20, pady=20, sticky='en')
        # self.entry.bind('<Return>', self.btn_transcribe_on_click)

        button = tk.Button(self, text="Transcribe", font=("Times New Roman", 16),
                           command=self.btn_transcribe_on_click)
        button.grid(column=4, row=2, padx=20, pady=20, sticky='wn')

        label = tk.Label(self, text="Old GRU", font=("Times New Roman", 16))
        label.grid(column=1, row=4, columnspan=2, rowspan=1, padx=20, pady=20, sticky='n')

        label = tk.Label(self, text="New GRU", font=("Times New Roman", 16))
        label.grid(column=3, row=4, columnspan=2, rowspan=1, padx=20, pady=20, sticky='n')

        label = tk.Label(self, text="Baseline", font=("Times New Roman", 16))
        label.grid(column=5, row=4, columnspan=1, rowspan=1, padx=20, pady=20, sticky='n')

        self.entry_old_gru = tk.Entry(self, font=("Times New Roman", 16))
        # self.entry_old_gru.config(state="disabled")
        self.entry_old_gru.grid(column=1, row=5, columnspan=2, rowspan=1, padx=20, pady=20, sticky='n')

        self.entry_new_gru = tk.Entry(self, font=("Times New Roman", 16))
        # self.entry_new_gru.config(state="disabled")
        self.entry_new_gru.grid(column=3, row=5, columnspan=2, rowspan=1, padx=20, pady=20, sticky='n')

        self.entry_baseline = tk.Entry(self, font=("Times New Roman", 16))
        # self.entry_baseline.config(state="disabled")
        self.entry_baseline.grid(column=5, row=5, columnspan=1, rowspan=1, padx=20, pady=20, sticky='n')

        img_path = 'docs/gru_diagram_adresa.png'
        image = Image.open(img_path)
        photo = ImageTk.PhotoImage(self.resize_img(image, w=screenwidth // 3))

        label = tk.Label(self, image = photo)
        label.image = photo
        label.grid(column=1, row=6, columnspan=2, rowspan=2, padx=20, pady=20, sticky='n')

        label = tk.Label(self, image = photo)
        label.image = photo
        label.grid(column=3, row=6, columnspan=2, rowspan=2, padx=20, pady=20, sticky='n')

        self.mainloop()

    def btn_transcribe_on_click(self):
        word = self.entry.get()
        self.entry.delete(0, tk.END)

        if word:
            print(word)

        gru_old, gru_new, baseline = self.inference_engine(word)
        if gru_old:
            print(f'old GRU:     {gru_old}')
            self.entry_old_gru.delete(0, tk.END)
            self.entry_old_gru.insert(0, gru_old)
        if gru_new:
            print(f'new GRU: {gru_new}')
            self.entry_new_gru.delete(0, tk.END)
            self.entry_new_gru.insert(0, gru_new)
        if baseline:
            print(f'baseline:    {baseline}')
            self.entry_baseline.delete(0, tk.END)
            self.entry_baseline.insert(0, baseline)
    
    def resize_img(self, image, w=None, h=None):
        if not w and not h:
            return image
        
        if not w:
            w = int(image.width * h / image.height)
        if not h:
            h = int(image.height * w / image.width)
        
        return image.resize((w, h))


if __name__ == '__main__':
    main()