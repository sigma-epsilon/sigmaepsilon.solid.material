from enum import Enum, unique


@unique
class MaterialModelType(Enum):
    DEFAULT = 0
    MEMBRANE = 1
    PLATE_UFLYAND_MINDLIN = 2
    PLATE_KIRCHHOFF_LOVE = 3
    SHELL_UFLYAND_MINDLIN = 4
    SHELL_KIRCHHOFF_LOVE = 5
    UNDEFINED = -1

    @property
    def number_of_material_stress_variables(self) -> int:
        if self == MaterialModelType.DEFAULT:
            return 6
        elif self == MaterialModelType.MEMBRANE:
            return 3
        elif self in (
            MaterialModelType.PLATE_UFLYAND_MINDLIN,
            MaterialModelType.PLATE_KIRCHHOFF_LOVE,
            MaterialModelType.SHELL_UFLYAND_MINDLIN,
            MaterialModelType.SHELL_KIRCHHOFF_LOVE,
        ):
            return 5
        elif self == MaterialModelType.UNDEFINED:
            return None
        else:  # pragma: no cover
            raise NotImplementedError(
                (
                    f"This is not implemented for model type {self}. "
                    "Raise an issue if this is important to you or get involved "
                    "and make a pull request yourself."
                )
            )

    @property
    def number_of_material_stress_components(self) -> int:
        return self.number_of_material_stress_variables

    @property
    def number_of_stress_variables(self) -> int:
        if self == MaterialModelType.DEFAULT:
            return 6
        elif self == MaterialModelType.MEMBRANE:
            return 3
        elif self in (
            MaterialModelType.PLATE_UFLYAND_MINDLIN,
            MaterialModelType.PLATE_KIRCHHOFF_LOVE,
        ):
            return 5
        elif self in (
            MaterialModelType.SHELL_UFLYAND_MINDLIN,
            MaterialModelType.SHELL_KIRCHHOFF_LOVE,
        ):
            return 8
        elif self == MaterialModelType.UNDEFINED:
            return None
        else:  # pragma: no cover
            raise NotImplementedError(
                (
                    f"This is not implemented for model type {self}. "
                    "Raise an issue if this is important to you or get involved "
                    "and make a pull request yourself."
                )
            )

    @property
    def number_of_stress_components(self) -> int:
        return self.number_of_stress_variables
