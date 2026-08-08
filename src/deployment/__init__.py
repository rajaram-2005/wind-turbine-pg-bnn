"""Deployment utilities: ONNX export and C++ edge inference.

Modules:
    export_onnx: Export the PG-BNN to ONNX with mean + variance heads.
    cpp_inference/: C++ ONNX Runtime engine for edge devices (see CMakeLists).
"""

from src.deployment.export_onnx import export_bnn_to_onnx, validate_onnx_export

__all__ = ["export_bnn_to_onnx", "validate_onnx_export"]
