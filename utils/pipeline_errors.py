"""
Custom exception hierarchy for the ML pipeline.
Provides standardized error tracking and serialization for logs.
"""


# =========================================================================
# ERROR REGISTRY
# =========================================================================

SYSTEM_ERRORS = {
    "SYS-101": "Required features missing from input dataset",
    "SYS-102": "Schema mapping violation detected",
    "SYS-103": "Data quality bounds check failed",
    "SYS-201": "Failed to locate configuration file",
    "SYS-202": "Configuration parsing error",
    "SYS-301": "Model artifact not found in registry",
    "SYS-302": "Inference execution failed",
    "SYS-401": "Invalid user segment or optimization goal",
}


# =========================================================================
# BASE EXCEPTION
# =========================================================================

class MLSystemFault(Exception):
    """
    Root exception for all custom pipeline errors.
    """
    def __init__(self, code: str, context_msg: str = "") -> None:
        self.code = code
        self.base_msg = SYSTEM_ERRORS.get(code, "Unregistered system error")
        self.context_msg = context_msg

        self.formatted_message = f"[{self.code}] {self.base_msg}"
        if self.context_msg:
            self.formatted_message += f" | Context: {self.context_msg}"

        super().__init__(self.formatted_message)

    def serialize(self) -> dict[str, str]:
        """Serializes error details for JSON loggers."""
        return {
            "error_code": self.code,
            "error_type": self.__class__.__name__,
            "message": self.base_msg,
            "context": self.context_msg
        }


# =========================================================================
# SPECIFIC EXCEPTIONS
# =========================================================================

class ConfigurationFault(MLSystemFault):
    """Raised during settings load or validation failures."""
    def __init__(self, context_msg: str, code: str = "SYS-202") -> None:
        super().__init__(code, context_msg)


class SchemaValidationFault(MLSystemFault):
    """Raised when incoming data violates expected types or columns."""
    def __init__(self, context_msg: str, code: str = "SYS-101") -> None:
        super().__init__(code, context_msg)
