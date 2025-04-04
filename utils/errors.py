# ct2fea/utils/errors.py
from typing import Optional, Dict, Any
import logging
import datetime
import json
from pathlib import Path


class CT2FEAError(Exception):
    """Base exception class for CT2FEA errors"""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.details = details
        self.context = context or {}
        self.logger = logging.getLogger(__name__)

        # Log error with context for better debugging
        if context:
            self.logger.error(f"{message} - {details} - Context: {context}")
        else:
            self.logger.error(f"{message} - {details}")

        super().__init__(message)


class CTDataError(CT2FEAError):
    """Raised for CT data loading/processing errors"""

    pass


class SegmentationError(CT2FEAError):
    """Raised for segmentation-related errors"""

    pass


class MeshingError(CT2FEAError):
    """Raised for mesh generation/quality errors"""

    pass


class MaterialError(CT2FEAError):
    """Raised for material property calculation errors"""

    pass


class AccelerationError(CT2FEAError):
    """Base class for acceleration-related errors"""

    pass


class CUDAError(AccelerationError):
    """Raised for CUDA-specific errors"""

    pass


class OpenCLError(AccelerationError):
    """Raised for OpenCL-specific errors"""

    pass


class ExportError(CT2FEAError):
    """Raised for file export errors"""

    pass


class ConfigError(CT2FEAError):
    """Raised for configuration validation errors"""

    pass


class RecoverableError(CT2FEAError):
    """Error that allows for recovery and resumption of pipeline"""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        checkpoint_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details, context)
        self.checkpoint_data = checkpoint_data or {}

    def save_checkpoint(self, output_dir: Path) -> Path:
        """Save checkpoint data to allow pipeline resumption

        Args:
            output_dir: Directory to save checkpoint

        Returns:
            Path to checkpoint file
        """
        checkpoint_path = output_dir / "pipeline_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(self.checkpoint_data, f, indent=2)

        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path


class CheckpointManager:
    """Manages pipeline checkpoints for error recovery"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.checkpoint_path = output_dir / "pipeline_checkpoint.json"

    def save_checkpoint(self, stage: str, data: Dict[str, Any]) -> None:
        """Save pipeline checkpoint

        Args:
            stage: Current pipeline stage
            data: Data to checkpoint
        """
        checkpoint = {
            "stage": stage,
            "timestamp": str(datetime.datetime.now()),
            "data": data,
        }

        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        self.logger.info(f"Saved checkpoint for stage '{stage}'")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint if available

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self) -> None:
        """Clear existing checkpoint"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            self.logger.info("Cleared pipeline checkpoint")


def handle_gpu_error(e: Exception) -> None:
    """Handle GPU-related errors with appropriate fallback

    Args:
        e: Original exception

    Raises:
        AccelerationError: If no fallback is available
    """
    import logging

    logger = logging.getLogger(__name__)

    if isinstance(e, (CUDAError, ImportError)):
        logger.warning("CUDA acceleration failed, trying OpenCL fallback")
        try:
            from ..acceleration.opencl_utils import OpenCLManager

            return OpenCLManager()
        except (ImportError, OpenCLError) as ocl_e:
            logger.error("OpenCL fallback also failed")
            raise AccelerationError(
                "GPU acceleration unavailable",
                f"CUDA error: {str(e)}\nOpenCL error: {str(ocl_e)}",
            )
    raise AccelerationError(
        "GPU acceleration failed", str(e), {"original_exception": str(e)}
    )
