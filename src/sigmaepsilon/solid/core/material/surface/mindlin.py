from typing import Callable, Iterable, Tuple
from numbers import Number
import warnings

from numpy import ndarray, ascontiguousarray as ascont
import numpy as np
from numpy.linalg import inv

from .surface import SurfaceSection, SurfaceLayer
from ..utils import mindlin_elastic_material_stiffness_matrix
from ..utils.mindlin import (
    _get_shell_material_stiffness_matrix,
    shell_rotation_matrix,
)
from ..warnings import SigmaEpsilonMaterialWarning

__all__ = ["MindlinShellSection", "MindlinPlateSection"]


class MindlinShellLayer(SurfaceLayer):
    """
    A class for layers of Uflyand-Mindlin shells.
    """

    __loc__ = [-1.0, 0.0, 1.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Shear factors have to be multiplied by shear force to obtain shear
        # stress. They are determined externaly at discrete points, which
        # are later used for interpolation.
        self._shear_factors_x = np.array([0.0, 0.0, 0.0])
        self._shear_factors_y = np.array([0.0, 0.0, 0.0])

        # polinomial coefficients for shear factor interpoaltion
        self._sfx = None
        self._sfy = None

    def material_elastic_stiffness_matrix(self) -> ndarray:
        """
        Returns and stores the transformed material stiffness matrix.
        """
        Cm = _get_shell_material_stiffness_matrix(self.material)
        T = self.rotation_matrix()
        R = np.diag([1.0, 1.0, 2.0, 2.0, 2.0])
        self.material = inv(T) @ Cm @ R @ T @ inv(R)
        return self.material

    def rotation_matrix(self) -> ndarray:
        """
        Returns the transformation matrix of the layer.
        """
        return shell_rotation_matrix(self.angle, rad=False)

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        Returns the uncorrected stiffness contribution to the layer.
        """
        C = self.material_elastic_stiffness_matrix()
        tmin = self._tmin
        tmax = self._tmax
        A = C[:3, :3] * (tmax - tmin)
        B = (1 / 2) * C[:3, :3] * (tmax**2 - tmin**2)
        D = (1 / 3) * C[:3, :3] * (tmax**3 - tmin**3)
        S = C[3:, 3:] * (tmax - tmin)
        ABDS = np.zeros([8, 8], dtype=float)
        ABDS[0:3, 0:3] = A
        ABDS[0:3, 3:6] = B
        ABDS[3:6, 0:3] = B
        ABDS[3:6, 3:6] = D
        ABDS[6:8, 6:8] = S
        return ABDS

    def _compile_shear_factors(self) -> None:
        """
        Prepares data for continuous interpolation of shear factors. Should
        be called if shear factors are already set.
        """
        coeff_inv = inv(np.array([[1, z, z**2] for z in self._zi]))
        self._sfx = np.matmul(coeff_inv, self._shear_factors_x)
        self._sfy = np.matmul(coeff_inv, self._shear_factors_y)

    def _loc_to_shear_factors(self, loc: Number) -> Tuple[float, float]:
        """
        Returns shear factor for local z direction by quadratic interpolation.
        Local coordinate is expected between -1 and 1.
        """
        z = self._loc_to_z(loc)
        monoms = np.array([1, z, z**2])
        return np.dot(monoms, self._sfx), np.dot(monoms, self._sfy)

    def approximator(self, data: Iterable[Number]) -> Callable[[Number], float]:
        """
        Returns a function that can be used for approximations thorugh
        the thickness.
        """
        z0, z1, z2 = self._zi
        z = np.array([[1, z0, z0**2], [1, z1, z1**2], [1, z2, z2**2]])
        a, b, c = inv(z) @ np.array(data)
        return lambda z: a + b * z + c * z**2


class MindlinShellSection(SurfaceSection[MindlinShellLayer]):
    layer_class = MindlinShellLayer

    @staticmethod
    def Material(**kwargs) -> ndarray:
        return mindlin_elastic_material_stiffness_matrix(**kwargs)

    def _elastic_stiffness_matrix(self, out: ndarray) -> None:
        """
        Returns the stiffness matrix of the shell.

        Returns
        -------
        numpy.ndarray
            The ABDS matrix of the shell.
        """
        super()._elastic_stiffness_matrix(out)

        ABDS = out

        layers: Iterable[MindlinShellLayer] = self.layers
        A11 = ABDS[0, 0]
        B11 = ABDS[0, 3]
        D11 = ABDS[3, 3]
        S55 = ABDS[6, 6]
        A22 = ABDS[1, 1]
        B22 = ABDS[1, 4]
        D22 = ABDS[4, 4]
        S44 = ABDS[7, 7]

        ABDS_inv = np.linalg.inv(ABDS[:6, :6])
        alpha_x = ABDS_inv[0, 3]
        beta_x = ABDS_inv[3, 3]
        alpha_y = ABDS_inv[1, 4]
        beta_y = ABDS_inv[4, 4]

        eta_x = 1 / (A11 * D11 - B11**2)
        eta_y = 1 / (A22 * D22 - B22**2)
        alpha_x = -B11 * eta_x
        beta_x = A11 * eta_x
        alpha_y = -B22 * eta_y
        beta_y = A22 * eta_y

        # Create shear factors. These need to be multiplied with the shear
        # force in order to obtain shear stress at a given height. Since the
        # shear stress distribution is of 2nd order, the factors are
        # determined at 3 locations per layer.
        for i, layer in enumerate(layers):
            zi = layer._zi
            Exi = layer._material[0, 0]
            Eyi = layer._material[1, 1]

            # first point through the thickness
            layer._shear_factors_x[0] = layers[i - 1]._shear_factors_x[-1]
            layer._shear_factors_y[0] = layers[i - 1]._shear_factors_y[-1]

            # second point through the thickness
            layer._shear_factors_x[1] = layer._shear_factors_x[0] - Exi * (
                0.5 * (zi[1] ** 2 - zi[0] ** 2) * beta_x + (zi[1] - zi[0]) * alpha_x
            )
            layer._shear_factors_y[1] = layer._shear_factors_y[0] - Eyi * (
                0.5 * (zi[1] ** 2 - zi[0] ** 2) * beta_y + (zi[1] - zi[0]) * alpha_y
            )

            # third point through the thickness
            layer._shear_factors_x[2] = layer._shear_factors_x[0] - Exi * (
                0.5 * (zi[2] ** 2 - zi[0] ** 2) * beta_x + (zi[2] - zi[0]) * alpha_x
            )
            layer._shear_factors_y[2] = layer._shear_factors_y[0] - Eyi * (
                0.5 * (zi[2] ** 2 - zi[0] ** 2) * beta_y + (zi[2] - zi[0]) * alpha_y
            )

        # remove numerical junk from the end
        layers[-1]._shear_factors_x[-1] = 0.0
        layers[-1]._shear_factors_y[-1] = 0.0

        # prepare data for interpolation of shear stresses in a layer
        for layer in layers:
            layer._compile_shear_factors()

        # potential energy using constant stress distribution
        # and unit shear force
        pot_c_x = 0.5 / S55
        pot_c_y = 0.5 / S44

        # positions and weights of Gauss-points
        gP = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
        gW = np.array([5 / 9, 8 / 9, 5 / 9])

        # potential energy using parabolic stress distribution
        # and unit shear force
        pot_p_x, pot_p_y = 0.0, 0.0
        for layer in layers:
            dJ = 0.5 * (layer._tmax - layer._tmin)
            Gxi = layer._material[-2, -2]
            Gyi = layer._material[-1, -1]
            for loc, weight in zip(gP, gW):
                sfx, sfy = layer._loc_to_shear_factors(loc)
                pot_p_x += 0.5 * (sfx**2) * dJ * weight / Gxi
                pot_p_y += 0.5 * (sfy**2) * dJ * weight / Gyi
        kx = pot_c_x / pot_p_x
        ky = pot_c_y / pot_p_y

        ABDS[6, 6] = kx * S55
        ABDS[7, 7] = ky * S44
        self._shear_correction_factor_x = kx
        self._shear_correction_factor_y = ky


class MindlinPlateSection(MindlinShellSection):
    """
    A class for Uflyand-Mindlin plates.
    """

    @MindlinShellSection.eccentricity.setter
    def eccentricity(self, _: Number) -> None:
        raise Exception(
            "Plates can't have eccentricity, consider using a shell instead."
        )

    def elastic_stiffness_matrix(self, tol: Number = 1e-8) -> ndarray:
        """
        Assembles and returns the stiffness matrix.
        """
        self._set_layers()
        ABDS = np.zeros(self.layer_class.__shape__)
        self._elastic_stiffness_matrix(ABDS)

        D = ABDS[3:6, 3:6]
        B = ABDS[:3, 3:6]
        sumB = np.sum(np.abs(B))
        if sumB > tol:
            warnings.warn(
                "It seems that handling this section as a plate only would result in "
                "missing out on the effect of bending-extension coupling. "
                "It is suggested to use a shell instead.",
                SigmaEpsilonMaterialWarning
            )

        self._ABDS = ascont(D)
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS
    