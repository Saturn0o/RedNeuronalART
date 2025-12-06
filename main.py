import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

from art_network import ART
from image_processor import process_image

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal ART")
        self.root.geometry("1000x600")

        self.image_path = None
        self.grid_size = 28
        self.canvas_size = 308
        self.cell_size = self.canvas_size / self.grid_size
        self.current_pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        self.input_size = self.grid_size * self.grid_size
        self.max_categories = 50
        self.vigilance = tk.DoubleVar(value=0.7)
        self.art_net = ART(n=self.input_size, m=self.max_categories, rho=self.vigilance.get())

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_frame, width=250)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vigilance_frame = ttk.LabelFrame(controls_frame, text="Parámetro de Vigilancia (ρ)")
        vigilance_frame.pack(fill=tk.X, pady=10, ipady=5)
        
        self.vigilance_label = ttk.Label(vigilance_frame, text=f"{self.vigilance.get():.1f}")
        self.vigilance_label.pack()
        
        scale_vigilance = ttk.Scale(vigilance_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.vigilance, command=self.update_vigilance)
        scale_vigilance.pack(fill=tk.X, padx=5, pady=5)

        btn_train = ttk.Button(controls_frame, text="Entrenar Red", command=self.train_network)
        btn_train.pack(fill=tk.X, pady=5)

        btn_test = ttk.Button(controls_frame, text="Probar Patrón", command=self.test_pattern)
        btn_test.pack(fill=tk.X, pady=5)

        btn_reset = ttk.Button(controls_frame, text="Resetear Red", command=self.reset_network)
        btn_reset.pack(fill=tk.X, pady=20)

        original_frame = ttk.LabelFrame(images_frame, text="Imagen Original")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lbl_original_image = ttk.Label(original_frame)
        self.lbl_original_image.pack(padx=10, pady=10)

        processed_frame = ttk.LabelFrame(images_frame, text="Patrón Editable (28x28)")
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.grid_canvas = tk.Canvas(processed_frame, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.grid_canvas.pack(pady=5)
        self.grid_canvas.bind("<Button-1>", self._handle_grid_click)

        btn_load = ttk.Button(processed_frame, text="Cargar Imagen a la Cuadrícula", command=self.load_image)
        btn_load.pack(pady=5)
        
        self._setup_learned_patterns_display(main_frame)

        self._draw_grid()
        self._update_learned_patterns_display()


    def _setup_learned_patterns_display(self, parent):
        patterns_frame = ttk.LabelFrame(parent, text="Patrones Aprendidos")
        patterns_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.patterns_canvas = tk.Canvas(patterns_frame, borderwidth=0, background="#ffffff")
        self.patterns_list_frame = ttk.Frame(self.patterns_canvas, padding=(5, 5))
        
        scrollbar = ttk.Scrollbar(patterns_frame, orient="vertical", command=self.patterns_canvas.yview)
        self.patterns_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.patterns_canvas.pack(side="left", fill="both", expand=True)
        
        self.canvas_frame_id = self.patterns_canvas.create_window((0, 0), window=self.patterns_list_frame, anchor="nw")
        
        self.patterns_list_frame.bind("<Configure>", self._on_frame_configure)
        self.patterns_canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_frame_configure(self, event=None):
        self.patterns_canvas.configure(scrollregion=self.patterns_canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        self.patterns_canvas.itemconfig(self.canvas_frame_id, width=self.patterns_canvas.winfo_width())

    def _update_learned_patterns_display(self):
        for widget in self.patterns_list_frame.winfo_children():
            widget.destroy()

        prototypes = self.art_net.get_committed_prototypes()

        if not prototypes:
            ttk.Label(self.patterns_list_frame, text="Ningún patrón aprendido.").pack()
            return

        for i, (neuron_index, pattern_flat) in enumerate(prototypes):
            pattern_2d = pattern_flat.reshape((self.grid_size, self.grid_size))
            
            neuron_frame = ttk.Frame(self.patterns_list_frame, padding=(5,5))
            neuron_frame.pack(fill=tk.X, expand=True, pady=5)

            ttk.Label(neuron_frame, text=f"Neurona #{neuron_index + 1}", font=("", 9, "bold")).pack()
            
            pattern_canvas_size = 84
            p_canvas = tk.Canvas(neuron_frame, width=pattern_canvas_size, height=pattern_canvas_size, bg="white")
            p_canvas.pack()
            
            cell_p_size = pattern_canvas_size / self.grid_size
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    color = "black" if pattern_2d[r, c] == 1 else "white"
                    x1, y1 = c * cell_p_size, r * cell_p_size
                    x2, y2 = x1 + cell_p_size, y1 + cell_p_size
                    p_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="lightgrey")


    def _draw_grid(self):
        self.grid_canvas.delete("all")
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = "black" if self.current_pattern[r, c] == 1 else "white"
                self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="grey")

    def _handle_grid_click(self, event):
        c = int(event.x // self.cell_size)
        r = int(event.y // self.cell_size)
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            self.current_pattern[r, c] = 1 - self.current_pattern[r, c]
            self._draw_grid()

    def _display_image(self, label, image_pil, max_size=(300, 300)):
        if image_pil is None:
            label.config(image='')
            label.image = None
            return
        
        image_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image_pil)
        label.config(image=photo)
        label.image = photo

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not path:
            return

        self.image_path = path
        try:
            pattern_flat, _ = process_image(self.image_path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo procesar la imagen.\nError: {e}")
            return


        if pattern_flat is None:
            messagebox.showerror("Error", "No se pudo procesar la imagen.")
            return

        self.current_pattern = pattern_flat.reshape((self.grid_size, self.grid_size))
        self._draw_grid()
        
        original_image = Image.open(self.image_path)
        self._display_image(self.lbl_original_image, original_image)

    def update_vigilance(self, value):
        val = round(float(value), 1)
        self.vigilance.set(val)
        self.art_net.set_vigilance(val)
        self.vigilance_label.config(text=f"{val:.1f}")

    def _get_flat_pattern(self):
        if self.current_pattern is None:
            return None
        return self.current_pattern.flatten()

    def train_network(self):
        flat_pattern = self._get_flat_pattern()
        if flat_pattern is None or np.sum(flat_pattern) == 0:
            messagebox.showwarning("Advertencia", "No hay un patrón en la cuadrícula para entrenar.")
            return

        neuron_index, is_new = self.art_net.train(flat_pattern)

        if neuron_index == -1:
            msg = "No hay más neuronas disponibles en la red."
        elif is_new:
            msg = f"Nuevo patrón aprendido.\nAlmacenado en la Neurona #{neuron_index + 1}."
            self._update_learned_patterns_display()
        else:
            msg = f"El patrón coincide con uno existente.\nCategoría reconocida: Neurona #{neuron_index + 1}."
        
        messagebox.showinfo("Resultado del Entrenamiento", msg)


    def test_pattern(self):
        flat_pattern = self._get_flat_pattern()
        if flat_pattern is None or np.sum(flat_pattern) == 0:
            messagebox.showwarning("Advertencia", "No hay un patrón en la cuadrícula para probar.")
            return

        neuron_index = self.art_net.test(flat_pattern)

        if neuron_index != -1:
            msg = f"El patrón coincide con la Neurona #{neuron_index + 1}."
        else:
            msg = "El patrón no coincide con ninguna categoría aprendida."
        
        messagebox.showinfo("Resultado de la Prueba", msg)


    def reset_network(self):
        if messagebox.askyesno("Confirmar", "¿Estás seguro de que quieres resetear la red? Se perderán todos los patrones aprendidos."):
            self.art_net.reset()
            self.current_pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            self._draw_grid()
            self._display_image(self.lbl_original_image, None)
            self._update_learned_patterns_display()
            messagebox.showinfo("Información", "La red ha sido reseteada a su estado inicial.")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use('clam') 
    app = App(root)
    root.mainloop()