# ct2fea/io/gui.py
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from pathlib import Path
import logging
from typing import Optional, Dict
import json
from ...config import Config


class ProgressDialog:
    def __init__(self, parent: tk.Tk, title: str):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()

        # Center window
        window_width = 300
        window_height = 100
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.top.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Progress bar
        self.progress = ttk.Progressbar(
            self.top, orient="horizontal", length=250, mode="determinate"
        )
        self.progress.pack(pady=10)

        # Status label
        self.status = ttk.Label(self.top, text="Initializing...")
        self.status.pack(pady=5)

        self.top.protocol("WM_DELETE_WINDOW", self._on_closing)

    def update(self, value: float, status: str):
        """Update progress and status text"""
        self.progress["value"] = value
        self.status["text"] = status
        self.top.update()

    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to cancel the operation?"):
            self.top.destroy()


class ConfigDialog:
    def __init__(self, parent: tk.Tk):
        self.top = tk.Toplevel(parent)
        self.top.title("CT2FEA Configuration")
        self.top.transient(parent)
        self.top.grab_set()

        # Initialize variables
        self.input_folder = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.voxel_size = tk.DoubleVar(value=10.0)
        self.material_model = tk.StringVar(value="linear")
        self.use_gpu = tk.BooleanVar(value=True)
        self.use_parallel = tk.BooleanVar(value=True)
        self.n_jobs = tk.StringVar(value="auto")  # auto = use all cores
        self.enable_interactive = tk.BooleanVar(value=True)
        self.result = None

        self._create_widgets()
        self._load_last_config()

    def _create_widgets(self):
        """Create dialog widgets"""
        notebook = ttk.Notebook(self.top)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Input/Output tab
        io_frame = ttk.Frame(notebook)
        notebook.add(io_frame, text="Input/Output")

        ttk.Button(
            io_frame, text="Select Input Folder", command=self._select_input
        ).pack(fill="x", pady=2)
        ttk.Entry(io_frame, textvariable=self.input_folder).pack(fill="x", pady=2)

        ttk.Button(
            io_frame, text="Select Output Directory", command=self._select_output
        ).pack(fill="x", pady=2)
        ttk.Entry(io_frame, textvariable=self.output_dir).pack(fill="x", pady=2)

        # Processing tab
        proc_frame = ttk.Frame(notebook)
        notebook.add(proc_frame, text="Processing")

        ttk.Label(proc_frame, text="Voxel Size (μm):").pack(anchor="w")
        ttk.Entry(proc_frame, textvariable=self.voxel_size).pack(fill="x", pady=2)

        # Material model
        ttk.Label(proc_frame, text="Material Model:").pack(anchor="w")
        models = ["linear", "plasticity", "hyperelastic"]
        ttk.OptionMenu(proc_frame, self.material_model, models[0], *models).pack(
            fill="x", pady=2
        )

        # Performance tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")

        # GPU acceleration
        ttk.Checkbutton(
            perf_frame, text="Use GPU Acceleration", variable=self.use_gpu
        ).pack(anchor="w", pady=2)

        # Parallel processing
        parallel_frame = ttk.LabelFrame(
            perf_frame, text="Parallel Processing", padding=5
        )
        parallel_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            parallel_frame,
            text="Enable Parallel Processing",
            variable=self.use_parallel,
        ).pack(anchor="w")

        cpu_frame = ttk.Frame(parallel_frame)
        cpu_frame.pack(fill="x")
        ttk.Label(cpu_frame, text="CPU Cores:").pack(side="left")
        ttk.Entry(cpu_frame, textvariable=self.n_jobs, width=10).pack(
            side="left", padx=5
        )
        ttk.Label(cpu_frame, text="(auto = use all cores)").pack(side="left")

        # Visualization tab
        vis_frame = ttk.Frame(notebook)
        notebook.add(vis_frame, text="Visualization")

        ttk.Checkbutton(
            vis_frame,
            text="Enable Interactive 3D Visualization",
            variable=self.enable_interactive,
        ).pack(anchor="w", pady=2)

        # Buttons
        button_frame = ttk.Frame(self.top)
        button_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(button_frame, text="OK", command=self._on_ok).pack(
            side="right", padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(
            side="right"
        )

    def _select_input(self):
        """Handle input folder selection"""
        folder = filedialog.askdirectory(title="Select CT Stack Folder")
        if folder:
            self.input_folder.set(folder)

    def _select_output(self):
        """Handle output directory selection"""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir.set(folder)

    def _on_ok(self):
        """Handle OK button"""
        try:
            # Convert n_jobs to integer or None
            n_jobs = (
                None if self.n_jobs.get().lower() == "auto" else int(self.n_jobs.get())
            )

            config = Config(
                input_folder=self.input_folder.get(),
                output_dir=self.output_dir.get(),
                voxel_size_um=self.voxel_size.get(),
                material_model=self.material_model.get(),
                use_gpu=self.use_gpu.get(),
                use_parallel=self.use_parallel.get(),
                n_jobs=n_jobs,
                visualization_dpi=300 if self.enable_interactive.get() else 0,
            )
            config.validate()
            self._save_config()
            self.result = config
            self.top.destroy()
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))

    def _on_cancel(self):
        """Handle Cancel button"""
        self.top.destroy()

    def _save_config(self):
        """Save current configuration"""
        config = {
            "input_folder": self.input_folder.get(),
            "output_dir": self.output_dir.get(),
            "voxel_size_um": self.voxel_size.get(),
            "material_model": self.material_model.get(),
            "use_gpu": self.use_gpu.get(),
            "use_parallel": self.use_parallel.get(),
            "n_jobs": self.n_jobs.get(),
            "enable_interactive": self.enable_interactive.get(),
        }

        try:
            with open(Path.home() / ".ct2fea_config", "w") as f:
                json.dump(config, f)
        except Exception as e:
            logging.warning(f"Could not save configuration: {e}")

    def _load_last_config(self):
        """Load last used configuration"""
        try:
            with open(Path.home() / ".ct2fea_config", "r") as f:
                config = json.load(f)

            self.input_folder.set(config.get("input_folder", ""))
            self.output_dir.set(config.get("output_dir", ""))
            self.voxel_size.set(config.get("voxel_size_um", 10.0))
            self.material_model.set(config.get("material_model", "linear"))
            self.use_gpu.set(config.get("use_gpu", True))
            self.use_parallel.set(config.get("use_parallel", True))
            self.n_jobs.set(config.get("n_jobs", "auto"))
            self.enable_interactive.set(config.get("enable_interactive", True))
        except Exception as e:
            logging.debug(f"Could not load last configuration: {e}")


def get_gui_inputs() -> Optional[Config]:
    """Show configuration dialog and get inputs

    Returns:
        Config object if OK was pressed, None if cancelled
    """
    root = tk.Tk()
    root.withdraw()

    dialog = ConfigDialog(root)
    root.wait_window(dialog.top)

    root.destroy()
    return dialog.result


def show_progress(title: str) -> ProgressDialog:
    """Create and show a progress dialog

    Args:
        title: Dialog title

    Returns:
        Progress dialog instance
    """
    root = tk.Tk()
    root.withdraw()
    return ProgressDialog(root, title)
