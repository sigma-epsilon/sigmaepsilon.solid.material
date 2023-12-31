from enum import Enum


class ModelType(Enum):
    DEFAULT = 0
    MEMBRANE = 1
    PLATE_UFLYAND_MINDLIN = 2
    PLATE_KIRCHHOFF_LOVE = 3
    SHELL_UFLYAND_MINDLIN = 4
    SHELL_KIRCHHOFF_LOVE = 5
