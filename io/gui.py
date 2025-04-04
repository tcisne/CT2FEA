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
        window_width = 400
        window_height = 200
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.top.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Main progress bar
        main_frame = ttk.LabelFrame(self.top, text="Overall Progress")
        main_frame.pack(fill="x", padx=10, pady=5)

        self.progress = ttk.Progressbar(
            main_frame, orient="horizontal", length=350, mode="determinate"
        )
        self.progress.pack(pady=5, padx=10, fill="x")

        # Status label
        self.status = ttk.Label(main_frame, text="Initializing...")
        self.status.pack(pady=5)

        # Stage progress
        stage_frame = ttk.LabelFrame(self.top, text="Current Stage")
        stage_frame.pack(fill="x", padx=10, pady=5)

        self.stage_progress = ttk.Progressbar(
            stage_frame, orient="horizontal", length=350, mode="determinate"
        )
        self.stage_progress.pack(pady=5, padx=10, fill="x")

        self.stage_status = ttk.Label(stage_frame, text="")
        self.stage_status.pack(pady=5)

        # Memory usage
        self.memory_label = ttk.Label(self.top, text="Memory usage: 0 MB")
        self.memory_label.pack(pady=5, anchor="w", padx=10)

        # Cancel button
        ttk.Button(self.top, text="Cancel", command=self._on_closing).pack(pady=10)

        self.top.protocol("WM_DELETE_WINDOW", self._on_closing)

        # For tracking cancellation
        self.cancelled = False

    def update(self, value: float, status: str):
        """Update overall progress and status text"""
        self.progress["value"] = value
        self.status["text"] = status
        self.top.update()

    def update_stage(self, value: float, status: str):
        """Update stage progress and status"""
        self.stage_progress["value"] = value
        self.stage_status["text"] = status
        self.top.update()

    def update_memory(self, usage_mb: float):
        """Update memory usage display"""
        if usage_mb > 1024:
            self.memory_label["text"] = f"Memory usage: {usage_mb / 1024:.2f} GB"
        else:
            self.memory_label["text"] = f"Memory usage: {usage_mb:.1f} MB"
        self.top.update()

    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to cancel the operation?"):
            self.cancelled = True
            self.top.destroy()

    def is_cancelled(self) -> bool:
        """Check if operation was cancelled"""
        return self.cancelled


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

        # New streaming options
        self.use_streaming = tk.BooleanVar(value=True)
        self.streaming_chunk_size = tk.IntVar(value=10)
        self.max_memory_usage = tk.DoubleVar(value=4.0)

        # Error recovery options
        self.enable_checkpoints = tk.BooleanVar(value=True)

        # Segmentation options
        self.adaptive_segmentation = tk.BooleanVar(value=True)
        self.pore_threshold = tk.DoubleVar(value=0.2)
        self.min_bone_size = tk.IntVar(value=100)
        self.min_pore_size = tk.IntVar(value=20)

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

        # Segmentation options
        seg_frame = ttk.LabelFrame(proc_frame, text="Segmentation", padding=5)
        seg_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            seg_frame,
            text="Use Adaptive Segmentation Parameters",
            variable=self.adaptive_segmentation,
        ).pack(anchor="w")

        pore_frame = ttk.Frame(seg_frame)
        pore_frame.pack(fill="x", pady=2)
        ttk.Label(pore_frame, text="Pore Threshold:").pack(side="left")
        ttk.Entry(pore_frame, textvariable=self.pore_threshold, width=10).pack(
            side="left", padx=5
        )

        bone_frame = ttk.Frame(seg_frame)
        bone_frame.pack(fill="x", pady=2)
        ttk.Label(bone_frame, text="Min Bone Size:").pack(side="left")
        ttk.Entry(bone_frame, textvariable=self.min_bone_size, width=10).pack(
            side="left", padx=5
        )

        pore_size_frame = ttk.Frame(seg_frame)
        pore_size_frame.pack(fill="x", pady=2)
        ttk.Label(pore_size_frame, text="Min Pore Size:").pack(side="left")
        ttk.Entry(pore_size_frame, textvariable=self.min_pore_size, width=10).pack(
            side="left", padx=5
        )

        # Material model
        ttk.Label(proc_frame, text="Material Model:").pack(anchor="w", pady=(10, 0))
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

        # Streaming options
        streaming_frame = ttk.LabelFrame(
            perf_frame, text="Memory Management", padding=5
        )
        streaming_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            streaming_frame,
            text="Enable Streaming Processing (for large datasets)",
            variable=self.use_streaming,
        ).pack(anchor="w")

        chunk_frame = ttk.Frame(streaming_frame)
        chunk_frame.pack(fill="x")
        ttk.Label(chunk_frame, text="Chunk Size (slices):").pack(side="left")
        ttk.Entry(chunk_frame, textvariable=self.streaming_chunk_size, width=10).pack(
            side="left", padx=5
        )

        memory_frame = ttk.Frame(streaming_frame)
        memory_frame.pack(fill="x", pady=2)
        ttk.Label(memory_frame, text="Max Memory Usage (GB):").pack(side="left")
        ttk.Entry(memory_frame, textvariable=self.max_memory_usage, width=10).pack(
            side="left", padx=5
        )

        # Error recovery options
        recovery_frame = ttk.LabelFrame(perf_frame, text="Error Recovery", padding=5)
        recovery_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            recovery_frame,
            text="Enable Checkpoints (for resuming after errors)",
            variable=self.enable_checkpoints,
        ).pack(anchor="w")

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
                # New streaming options
                use_streaming=self.use_streaming.get(),
                streaming_chunk_size=self.streaming_chunk_size.get(),
                max_memory_usage_gb=self.max_memory_usage.get(),
                # Error recovery options
                enable_checkpoints=self.enable_checkpoints.get(),
                # Segmentation options
                pore_threshold=self.pore_threshold.get(),
                adaptive_segmentation=self.adaptive_segmentation.get(),
                segmentation_params={
                    "min_bone_size": self.min_bone_size.get(),
                    "min_pore_size": self.min_pore_size.get(),
                    "pore_threshold": self.pore_threshold.get(),
                },
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
            # New streaming options
            "use_streaming": self.use_streaming.get(),
            "streaming_chunk_size": self.streaming_chunk_size.get(),
            "max_memory_usage": self.max_memory_usage.get(),
            # Error recovery options
            "enable_checkpoints": self.enable_checkpoints.get(),
            # Segmentation options
            "adaptive_segmentation": self.adaptive_segmentation.get(),
            "pore_threshold": self.pore_threshold.get(),
            "min_bone_size": self.min_bone_size.get(),
            "min_pore_size": self.min_pore_size.get(),
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

            # New streaming options
            self.use_streaming.set(config.get("use_streaming", True))
            self.streaming_chunk_size.set(config.get("streaming_chunk_size", 10))
            self.max_memory_usage.set(config.get("max_memory_usage", 4.0))

            # Error recovery options
            self.enable_checkpoints.set(config.get("enable_checkpoints", True))

            # Segmentation options
            self.adaptive_segmentation.set(config.get("adaptive_segmentation", True))
            self.pore_threshold.set(config.get("pore_threshold", 0.2))
            self.min_bone_size.set(config.get("min_bone_size", 100))
            self.min_pore_size.set(config.get("min_pore_size", 20))
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
