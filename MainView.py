# gui_polytech_search.py
import os
import sys
import re
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

DATA_DIR = "data"          # dossier contenant les .txt
ENGINE_SCRIPT = "tf-idf.py"  # le moteur existant
TOP_K = 10

RESULT_LINE_RE = re.compile(r"^\s*([0-9.]+)\s+(\S+\.txt)\s*$")

class PolytechSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polytech Search")
        self.geometry("1080x720")
        self.minsize(900, 600)

        self._build_ui()

    def _build_ui(self):
        # ===== Header =====
        header = tk.Frame(self)
        header.pack(fill="x", pady=16)

        title = tk.Label(
            header,
            text="Polytech Search",
            font=("Helvetica", 28, "bold")
        )
        title.pack()

        sub = tk.Label(
            header,
            text="Moteur de recherche – prototype",
            font=("Helvetica", 11)
        )
        sub.pack(pady=(4, 0))

        # ===== Search area =====
        search_area = tk.Frame(self)
        search_area.pack(pady=20)

        # Méthode (combo)
        method_label = tk.Label(search_area, text="Méthode :")
        method_label.grid(row=0, column=0, padx=(0, 8), sticky="e")

        self.method_var = tk.StringVar(value="TF-IDF")
        self.method_combo = ttk.Combobox(
            search_area,
            textvariable=self.method_var,
            values=["TF-IDF", "BM25 (à venir)", "Hybride (à venir)"],
            state="readonly",
            width=20
        )
        self.method_combo.grid(row=0, column=1, sticky="w")

        # Barre de recherche (style Google)
        self.query_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_area,
            textvariable=self.query_var,
            font=("Helvetica", 16),
            width=50,
            relief="solid",
            bd=1
        )
        self.search_entry.grid(row=1, column=0, columnspan=2, padx=8, pady=12, ipady=6)
        self.search_entry.bind("<Return>", lambda e: self.on_search())

        # Bouton Rechercher
        search_btn = tk.Button(
            search_area,
            text="Rechercher",
            font=("Helvetica", 12, "bold"),
            command=self.on_search
        )
        search_btn.grid(row=1, column=2, padx=(8, 0))

        # ===== Split main area: results (left) / preview (right) =====
        main = tk.PanedWindow(self, orient="horizontal", sashrelief="raised")
        main.pack(fill="both", expand=True, padx=12, pady=8)

        # Résultats
        left_frame = tk.Frame(main)
        main.add(left_frame, minsize=420)

        results_label = tk.Label(left_frame, text="Résultats", font=("Helvetica", 12, "bold"))
        results_label.pack(anchor="w")

        self.tree = ttk.Treeview(
            left_frame,
            columns=("score", "file"),
            show="headings",
            selectmode="browse",
            height=18
        )
        self.tree.heading("score", text="Score")
        self.tree.heading("file", text="Fichier")
        self.tree.column("score", width=100, anchor="center")
        self.tree.column("file", width=300, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=6)

        # Double-clic pour ouvrir le document
        self.tree.bind("<Double-1>", self.on_open_doc)

        # Prévisualisation du document
        right_frame = tk.Frame(main)
        main.add(right_frame)

        preview_label = tk.Label(right_frame, text="Aperçu du document", font=("Helvetica", 12, "bold"))
        preview_label.pack(anchor="w")

        self.preview = ScrolledText(right_frame, wrap="word", font=("Helvetica", 12))
        self.preview.pack(fill="both", expand=True, pady=6)

        # Barre d'état
        self.status_var = tk.StringVar(value="Prêt.")
        status = tk.Label(self, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=8, pady=(0, 6))

    def on_search(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showinfo("Info", "Veuillez saisir une requête.")
            return

        method = self.method_var.get()
        if method != "TF-IDF":
            messagebox.showinfo("Info", f"La méthode « {method} » sera ajoutée prochainement.\nUtilisation de TF-IDF pour l’instant.")
            # on continue en TF-IDF

        # Lancer tf-idf.py en mode 'search' et pousser la requête sur stdin
        cmd = [sys.executable, ENGINE_SCRIPT, "search","--query",query, "--k", str(TOP_K)]
        try:
            self.status_var.set("Recherche en cours…")
            self.update_idletasks()

            proc = subprocess.run(
                cmd,
                input=query + "\n",
                text=True,
                capture_output=True
            )
        except FileNotFoundError:
            messagebox.showerror("Erreur", f"Impossible de trouver {ENGINE_SCRIPT}. Place ce fichier dans le même dossier que l’interface.")
            self.status_var.set("Erreur.")
            return

        if proc.returncode != 0:
            # Affiche stdout/stderr pour déboguer
            messagebox.showerror("Erreur moteur",
                                 f"Code retour: {proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
            self.status_var.set("Erreur.")
            return

        # Parser les lignes de résultats (ignorer les barres de progression)
        results = []
        for line in proc.stdout.splitlines():
            m = RESULT_LINE_RE.match(line)
            if m:
                score = float(m.group(1))
                fname = m.group(2)
                if (score > 0):
                    results.append((score, fname))

        # Trier par score décroissant (par sécurité)
        results.sort(key=lambda x: x[0], reverse=True)

        # Afficher dans la Treeview
        for i in self.tree.get_children():
            self.tree.delete(i)
        for score, fname in results:
            self.tree.insert("", "end", values=(f"{score:.4f}", fname))

        self.status_var.set(f"{len(results)} résultat(s). Double-clique un fichier pour l’ouvrir.")

        # Effacer l’aperçu
        self.preview.delete("1.0", "end")

    def on_open_doc(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        _, fname = self.tree.item(sel[0], "values")
        path = os.path.join(DATA_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                txt = f.read()
        except FileNotFoundError:
            messagebox.showerror("Erreur", f"Fichier introuvable : {path}")
            return

        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", txt)
        self.status_var.set(f"Ouvert : {fname}")

if __name__ == "__main__":
    app = PolytechSearchApp()
    app.mainloop()
