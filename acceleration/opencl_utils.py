import pyopencl as cl
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from ..config import Config


@dataclass
class DeviceInfo:
    """OpenCL device information"""

    device: cl.Device
    type: str
    name: str
    compute_units: int
    global_memory: int
    local_memory: int
    max_work_group_size: int


class OpenCLError(Exception):
    """Base class for OpenCL-related errors"""

    pass


class DeviceNotFoundError(OpenCLError):
    """Raised when no suitable OpenCL device is found"""

    pass


class OpenCLManager:
    """Manage OpenCL context and acceleration operations"""

    def __init__(self, config: Optional[Config] = None):
        """Initialize OpenCL manager

        Args:
            config: Optional configuration object
        """
        self.ctx: Optional[cl.Context] = None
        self.queue: Optional[cl.CommandQueue] = None
        self.device: Optional[cl.Device] = None
        self.config = config
        self.logger = logging.getLogger(__name__)

    def initialize(self, device_type: str = "ALL") -> None:
        """Initialize OpenCL context with best available device

        Args:
            device_type: Type of device to use ("GPU", "CPU", or "ALL")

        Raises:
            DeviceNotFoundError: If no suitable device is found
            OpenCLError: For other OpenCL-related errors
        """
        try:
            # Get all available devices
            devices = self._get_devices(device_type)
            if not devices:
                raise DeviceNotFoundError(
                    f"No OpenCL devices found of type: {device_type}"
                )

            # Select best device
            self.device = self._select_best_device(devices)
            self.logger.info(f"Selected OpenCL device: {self.device.name}")

            # Create context and command queue
            self.ctx = cl.Context([self.device])
            self.queue = cl.CommandQueue(
                self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
            )

            # Log device capabilities
            self._log_device_info(self.device)

        except cl.Error as e:
            raise OpenCLError(f"OpenCL initialization failed: {str(e)}")
        except Exception as e:
            raise OpenCLError(
                f"Unexpected error during OpenCL initialization: {str(e)}"
            )

    def _get_devices(self, device_type: str) -> List[cl.Device]:
        """Get list of available OpenCL devices

        Args:
            device_type: Type of device to look for

        Returns:
            List of available devices
        """
        device_map = {
            "CPU": cl.device_type.CPU,
            "GPU": cl.device_type.GPU,
            "ALL": cl.device_type.ALL,
        }

        device_type_flag = device_map.get(device_type.upper(), cl.device_type.ALL)

        platforms = cl.get_platforms()
        devices = []

        for platform in platforms:
            try:
                platform_devices = platform.get_devices(device_type_flag)
                devices.extend(platform_devices)
            except cl.Error:
                continue

        return devices

    def _select_best_device(self, devices: List[cl.Device]) -> cl.Device:
        """Select the best device based on capabilities

        Args:
            devices: List of available devices

        Returns:
            Best available device
        """
        # Prefer GPU over CPU
        gpus = [d for d in devices if d.type == cl.device_type.GPU]
        if gpus:
            # Sort by compute units * max work group size
            return max(
                gpus,
                key=lambda d: (
                    d.max_compute_units
                    * d.max_work_group_size
                    * (2 if d.extensions.find("cl_khr_fp64") >= 0 else 1)
                ),
            )

        # Fallback to CPU with most compute units
        cpus = [d for d in devices if d.type == cl.device_type.CPU]
        if cpus:
            return max(cpus, key=lambda d: d.max_compute_units)

        # If no CPU/GPU, take first available device
        return devices[0]

    def _log_device_info(self, device: cl.Device) -> None:
        """Log detailed device information"""
        info = DeviceInfo(
            device=device,
            type="GPU" if device.type == cl.device_type.GPU else "CPU",
            name=device.name,
            compute_units=device.max_compute_units,
            global_memory=device.global_mem_size,
            local_memory=device.local_mem_size,
            max_work_group_size=device.max_work_group_size,
        )

        self.logger.info(
            f"OpenCL device details:\n"
            f"  Name: {info.name}\n"
            f"  Type: {info.type}\n"
            f"  Compute Units: {info.compute_units}\n"
            f"  Global Memory: {info.global_memory / (1024**2):.1f} MB\n"
            f"  Local Memory: {info.local_memory / 1024:.1f} KB\n"
            f"  Max Work Group Size: {info.max_work_group_size}"
        )

    def create_buffer(self, data: np.ndarray, mode: str = "r") -> cl.Buffer:
        """Create OpenCL buffer from numpy array

        Args:
            data: Input numpy array
            mode: Buffer mode ("r" for read-only, "w" for write-only)

        Returns:
            OpenCL buffer object
        """
        if self.ctx is None:
            self.initialize()

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")

        flags = cl.mem_flags.READ_ONLY if mode == "r" else cl.mem_flags.WRITE_ONLY
        try:
            return cl.Buffer(self.ctx, flags | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        except cl.Error as e:
            raise OpenCLError(f"Failed to create buffer: {str(e)}")

    def compile_kernel(self, source: str, kernel_name: str) -> cl.Kernel:
        """Compile OpenCL kernel from source

        Args:
            source: Kernel source code
            kernel_name: Name of kernel function

        Returns:
            Compiled kernel object
        """
        try:
            program = cl.Program(self.ctx, source).build()
            return getattr(program, kernel_name)
        except cl.Error as e:
            raise OpenCLError(f"Kernel compilation failed: {str(e)}")

    def run_kernel(
        self,
        kernel: cl.Kernel,
        global_size: Tuple[int, ...],
        local_size: Tuple[int, ...],
        *args: Any,
        wait_for: Optional[List[cl.Event]] = None,
    ) -> cl.Event:
        """Execute OpenCL kernel

        Args:
            kernel: Compiled kernel object
            global_size: Global work size
            local_size: Local work group size
            *args: Kernel arguments
            wait_for: Optional list of events to wait for

        Returns:
            Event object for kernel execution
        """
        try:
            return kernel(self.queue, global_size, local_size, *args, wait_for=wait_for)
        except cl.Error as e:
            raise OpenCLError(f"Kernel execution failed: {str(e)}")

    def get_preferred_work_group_size(
        self, kernel: cl.Kernel, device: Optional[cl.Device] = None
    ) -> int:
        """Get preferred work group size for kernel

        Args:
            kernel: Compiled kernel object
            device: Optional specific device

        Returns:
            Preferred work group size
        """
        if device is None:
            device = self.device

        try:
            return kernel.get_work_group_info(
                cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device
            )
        except cl.Error:
            # Fallback to device maximum
            return device.max_work_group_size
