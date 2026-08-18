"""
Type-safe enumerations for the ML pipeline.
Ensures IDE autocomplete and strict type checking across the system.
"""

from enum import Enum
from typing import Literal

# =========================================================================
# SEGMENT & GOAL ENUMERATIONS
# =========================================================================

class UserSegment(str, Enum):
    """
    Categorization of users based on their historical platform engagement.

    Attributes:
        POWER_USER: Extensive history -> Full complex feature set.
        CASUAL: Moderate history -> Standard feature set.
        GUEST: Little to no history -> Fallback / cold-start features.
    """
    POWER_USER = "power_user"
    CASUAL = "casual"
    GUEST = "guest"


class OptimizationGoal(str, Enum):
    """
    Defines the statistical thresholding strategy for the model output.
    """
    CAPTURE_ALL = "capture_all"     # Prioritizes Recall
    HIGH_SURETY = "high_surety"     # Prioritizes Precision
    BALANCED = "balanced"           # Standard F1 optimization


# =========================================================================
# TYPE ALIASES
# =========================================================================

SegmentLiteral = Literal["power_user", "casual", "guest"]
GoalLiteral = Literal["capture_all", "high_surety", "balanced"]
