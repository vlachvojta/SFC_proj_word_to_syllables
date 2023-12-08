import tkinter as tk
from tkinter import ttk


def on_click(entry):
    word = entry.get()
    entry.delete(0, tk.END)
    print(word)


def main():
    root = tk.Tk()

    root.title("Hyphenation program")
    width, height = 800, 600
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+0+0")
    # root.geometry(f"{width}x{height}")
    # root.geometry("widthxheight")
    # root.attributes('-fullscreen',True)

    notebook = ttk.Notebook(root)
    notebook.pack()

    frame_gru_complex = tk.Frame(notebook, width=w, height=h, bg='cyan')
    frame_gru_simple = tk.Frame(notebook, width=w, height=h, bg='white')

    frame_gru_complex.pack(fill='both')# , expand=1)
    frame_gru_simple.pack(fill='both')#, expand=1)

    notebook.add(frame_gru_complex, text='GRU Complex')
    notebook.add(frame_gru_simple, text='GRU Simple')


    label = tk.Label(frame_gru_complex, text="PyTorch GRU", font=("Times New Roman", 24))
    label.pack(padx=20, pady=20)

    entry = tk.Entry(frame_gru_complex, font=("Times New Roman", 16))
    entry.pack(padx=20, pady=20)

    button = tk.Button(frame_gru_complex, text="Transcribe", font=("Times New Roman", 16), command=lambda: on_click(entry))
    button.pack(padx=20, pady=20)


    root.mainloop()

if __name__ == '__main__':
    main()