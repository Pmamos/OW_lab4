import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from RSM import run_rsm_analysis
from UTA import uta_continuous, uta_discrete
import fuzzy_topsis
from topsis import topsis



class App:
    def __init__(self, master):
        self.master = master
        self.master.title("GUI_OW")
        
        self.alternatives_df = None
        self.classes_df = None
        
        # Ramka z przyciskami
        frame_top = tk.Frame(master)
        frame_top.pack(pady=10)
        
        self.btn_load_data = tk.Button(frame_top, text="Wczytaj dane z pliku", command=self.load_data)
        self.btn_load_data.pack(side=tk.LEFT, padx=5)
        
        # Combobox do wyboru metody
        self.method_var = tk.StringVar()
        self.method_box = ttk.Combobox(frame_top, textvariable=self.method_var, 
                                       values=["RSM (dyskr.)", "RSM (ciąg.)", "UTA (dyskretna)", "UTA (ciągła)", "Fuzzy TOPSIS"])
        self.method_box.set("Wybierz metodę...")
        self.method_box.pack(side=tk.LEFT, padx=5)
        
        self.btn_create_ranking = tk.Button(frame_top, text="Stwórz ranking", command=self.create_ranking)
        self.btn_create_ranking.pack(side=tk.LEFT, padx=5)
        
        # Ramka na tabelki
        frame_tables = tk.Frame(master)
        frame_tables.pack(pady=10, expand=True, fill='both')
        
        # Sekcja Alternatywy
        self.frame_alt = tk.LabelFrame(frame_tables, text="Alternatywy z kryteriami")
        self.frame_alt.pack(side=tk.LEFT, padx=10, expand=True, fill='both')
        
        self.frame_alt.config(width=400, height=200)
        self.frame_alt.pack_propagate(False)
        
        self.tree_alt = None
        
        # Sekcja Klasy
        self.frame_cls = tk.LabelFrame(frame_tables, text="Klasy")
        self.frame_cls.pack(side=tk.LEFT, padx=10, expand=True, fill='both')
        
        self.frame_cls.config(width=400, height=200)
        self.frame_cls.pack_propagate(False)

        self.tree_cls = ttk.Treeview(self.frame_cls, columns=["Nr","x","y","z"], show="headings", height=8)
        self.tree_cls.heading("Nr", text="Nr klasy")
        self.tree_cls.heading("x", text="x")
        self.tree_cls.heading("y", text="y")
        self.tree_cls.heading("z", text="z")

        self.tree_cls.column("Nr", width=60, anchor='center')  
        self.tree_cls.column("x", width=80, anchor='center')
        self.tree_cls.column("y", width=80, anchor='center')
        self.tree_cls.column("z", width=80, anchor='center')

        self.tree_cls.pack(padx=5, pady=5, expand=True, fill='both')
        
        # Tabela rankingu
        frame_rank = tk.LabelFrame(master, text="Stworzony ranking")
        frame_rank.pack(pady=10)
        
        self.tree_rank = ttk.Treeview(frame_rank, columns=["Nr","Score"], show="headings", height=8)
        self.tree_rank.heading("Nr", text="Nr Alternatywy")
        self.tree_rank.heading("Score", text="Wynik")
        self.tree_rank.pack(padx=5, pady=5)
        
        # Ramka na wykres
        self.frame_plot = tk.Frame(master)
        self.frame_plot.pack(pady=10)
        
        self.figure = Figure(figsize=(5,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame_plot)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack()
        
    def load_data(self):
        alt_file = filedialog.askopenfilename(title="Wybierz plik XLSX z alternatywami", filetypes=[("Excel files", "*.xlsx")])
        if not alt_file:
            return
        cls_file = filedialog.askopenfilename(title="Wybierz plik XLSX z klasami", filetypes=[("Excel files", "*.xlsx")])
        if not cls_file:
            return
        
        self.alternatives_df = pd.read_excel(alt_file, engine='openpyxl')
        self.classes_df = pd.read_excel(cls_file, engine='openpyxl')

        if self.tree_alt is not None:
            self.tree_alt.destroy()
        
        # Dynamiczna konfiguracja tree_alt
        all_columns = self.alternatives_df.columns.tolist()
        self.tree_alt = ttk.Treeview(self.frame_alt, columns=all_columns, show='headings', height=8)

        for col in all_columns:
            self.tree_alt.heading(col, text=col)

        self.tree_alt.column(all_columns[0], width=60, anchor='center')
        self.tree_alt.column(all_columns[1], width=100, anchor='w')
        for c in all_columns[2:]:
            self.tree_alt.column(c, width=70, anchor='center')

        for i, row in self.alternatives_df.iterrows():
            values = row.tolist()
            self.tree_alt.insert("", "end", values=values)

        self.tree_alt.pack(padx=5, pady=5, expand=True, fill='both')
        
        # Wyświetlenie klas
        for i in self.tree_cls.get_children():
            self.tree_cls.delete(i)
        for i, row in self.classes_df.iterrows():
            self.tree_cls.insert("", "end", values=[row["Nr klasy"], row["x"], row["y"], row["z"]])
        
        messagebox.showinfo("Informacja", "Pobrano dane")
        
    def create_ranking(self):
        method = self.method_var.get()
        if method == "Wybierz metodę...":
            messagebox.showerror("Błąd", "Wybierz metodę!")
            return
        print(method)
        
        # Jeśli metoda nie jest ciągła, potrzebujemy alternatyw i klas
        if method == "RSM (ciąg.)":
            ranking, score =  run_rsm_analysis()
        elif method == "RSM (dyskr.)":
            ranking, score = run_rsm_analysis()
        elif method == "UTA (dyskretna)":
            ranking, score = uta_discrete(points, A_star, theta)
        elif method == "UTA (ciągła)":
            ranking, score = uta_continuous(points, A_star, theta)
        elif method == "Fuzzy TOPSIS":
            ranking, score = fuzzy_topsis(data, weights, criteria)
        else:
            messagebox.showerror("Błąd", "Metoda nieobsługiwana.")
            return

        # Wyświetlenie rankingu
        for i in self.tree_rank.get_children():
            self.tree_rank.delete(i)
        
        for r, s in zip(ranking, score[ranking]):
            alt_nr = self.alternatives_df.iloc[r]["Nr alternatywy"]
            self.tree_rank.insert("", "end", values=[alt_nr, round(float(s),4)])
        
        
        messagebox.showinfo("Informacja", "Ranking utworzony!")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()