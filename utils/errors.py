# ct2fea/utils/errors.py
from typing import Optional


class CT2FEAError(Exception):
    """Base exception class for CT2FEA errors"""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
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
    raise AccelerationError("GPU acceleration failed", str(e))
