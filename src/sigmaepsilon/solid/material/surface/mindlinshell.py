from typing import Union, Optional, Iterable, Tuple, Callable
from numbers import Number

from numpy import ndarray
import numpy as np
from numpy.linalg import inv

from sigmaepsilon.math import atleastnd, atleast1d
from sigmaepsilon.math import to_range_1d

from ..utils.hmh import HMH_S_multi
from .surface import SurfaceSection, SurfaceLayer
from ..utils import mindlin_elastic_material_stiffness_matrix
from ..utils.mindlin import (
    _get_shell_material_stiffness_matrix,
    shell_rotation_matrix,
    z_to_shear_factors,
)
from ..utils.postproc import pproc_Mindlin_3D
from ..enums import MaterialModelType

__all__ = ["MindlinShellSection"]


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
        num_points_per_layer = len(self.__loc__)
        self._shear_factors_x = np.zeros((num_points_per_layer,), dtype=float)
        self._shear_factors_y = np.zeros((num_points_per_layer,), dtype=float)

        # polinomial coefficients for shear factor interpoaltion
        self._sfx = None
        self._sfy = None

    @property
    def shear_factors(self) -> ndarray:
        """
        Returns the shear factors of the layer. These factors multiplied with shear force'
        produce the shear stress according to Zsuravsykij's theory. If there are N number of
        evaluations per layer (the default is N=3), this returns an N by 2 matrix.

        Note
        ----
        It only returns valid data if it comes after a succesful stiffness matrix calculation,
        soin which case data from the remaining calculations is reused.
        """
        return np.stack([self._shear_factors_x, self._shear_factors_y], axis=1)

    def material_elastic_stiffness_matrix(self) -> ndarray:
        """
        Returns and stores the transformed material stiffness matrix.
        """
        Cm = _get_shell_material_stiffness_matrix(self.material_stiffness_matrix)
        T = self.rotation_matrix()
        R = np.diag([1.0, 1.0, 2.0, 2.0, 2.0])
        return inv(T) @ Cm @ R @ T @ inv(R)

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
        B = (1 / 2) * C[:3, :3] * (tmax ** 2 - tmin ** 2)
        D = (1 / 3) * C[:3, :3] * (tmax ** 3 - tmin ** 3)
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
        coeff_inv = inv(np.array([[1, z, z ** 2] for z in self._zi]))
        self._sfx = np.matmul(coeff_inv, self._shear_factors_x)
        self._sfy = np.matmul(coeff_inv, self._shear_factors_y)

    def _loc_to_shear_factors(self, loc: Number) -> Tuple[float, float]:
        """
        Returns shear factor for local z direction by quadratic interpolation.
        Local coordinate is expected between -1 and 1.
        """
        if self._sfx is None:
            self._compile_shear_factors()

        z = self._loc_to_z(loc)

        return z_to_shear_factors(z, self._sfx, self._sfy)

    def approximator(self, data: Iterable[Number]) -> Callable[[Number], float]:
        """
        Returns a function that can be used for approximations thorugh
        the thickness.
        """
        z0, z1, z2 = self._zi
        z = np.array([[1, z0, z0 ** 2], [1, z1, z1 ** 2], [1, z2, z2 ** 2]])
        a, b, c = inv(z) @ np.array(data)
        return lambda z: a + b * z + c * z ** 2


class MindlinShellSection(SurfaceSection[MindlinShellLayer]):
    """
    A class for Ufluand-Mindlin shells.

    Examples
    --------
    >>> from sigmaepsilon.solid.material import MindlinShellSection as Section
    >>> from sigmaepsilon.math.linalg import ReferenceFrame
    >>> from sigmaepsilon.solid.material import ElasticityTensor
    >>> from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix
    >>> ...
    >>> E = 2890.0
    >>> nu = 0.2
    >>> yield_strength = 2.0
    >>> thickness = 25.0
    >>> ...
    >>> hooke = elastic_stiffness_matrix(E=E, NU=nu)
    >>> frame = ReferenceFrame(dim=3)
    >>> tensor = ElasticityTensor(
    ...    hooke, frame=frame, tensorial=False, yield_strength=yield_strength
    ... )
    >>> ...
    >>> section = Section(
    ...     layers=[
    ...         Section.Layer(material=tensor, thickness=thickness / 3),
    ...         Section.Layer(material=tensor, thickness=thickness / 3),
    ...         Section.Layer(material=tensor, thickness=thickness / 3),
    ...     ]
    ... )
    >>> section.elastic_stiffness_matrix().shape
    (8, 8)
    """

    layer_class = MindlinShellLayer
    model_type = MaterialModelType.SHELL_UFLYAND_MINDLIN

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

        eta_x = 1 / (A11 * D11 - B11 ** 2)
        eta_y = 1 / (A22 * D22 - B22 ** 2)
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
            C = layer.material_stiffness_matrix
            Exi, Eyi = C[0, 0], C[1, 1]
            num_points_per_layer = len(layer.__loc__)

            assert num_points_per_layer > 0

            # first point through the thickness
            layer._shear_factors_x[0] = layers[i - 1]._shear_factors_x[-1]
            layer._shear_factors_y[0] = layers[i - 1]._shear_factors_y[-1]

            for j in range(1, num_points_per_layer):
                # second point through the thickness
                layer._shear_factors_x[j] = layer._shear_factors_x[0] - Exi * (
                    0.5 * (zi[j] ** 2 - zi[0] ** 2) * beta_x + (zi[j] - zi[0]) * alpha_x
                )
                layer._shear_factors_y[j] = layer._shear_factors_y[0] - Eyi * (
                    0.5 * (zi[j] ** 2 - zi[0] ** 2) * beta_y + (zi[j] - zi[0]) * alpha_y
                )

        # trim off numerical junk from the end
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
            C = layer.material_stiffness_matrix
            Gxi = C[-2, -2]
            Gyi = C[-1, -1]
            for loc, weight in zip(gP, gW):
                sfx, sfy = layer._loc_to_shear_factors(loc)
                pot_p_x += 0.5 * (sfx ** 2) * dJ * weight / Gxi
                pot_p_y += 0.5 * (sfy ** 2) * dJ * weight / Gyi
        kx = pot_c_x / pot_p_x
        ky = pot_c_y / pot_p_y

        ABDS[6, 6] = kx * S55
        ABDS[7, 7] = ky * S44
        self._shear_correction_factor_x = kx
        self._shear_correction_factor_y = ky

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        Assembles and returns the stiffness matrix.
        """
        self._set_layers()
        ABDS = np.zeros(self.layer_class.__shape__)
        self._elastic_stiffness_matrix(ABDS)

        self._ABDS = ABDS
        self._ABDS = (self._ABDS + self._ABDS.T) / 2
        self._SDBA = np.linalg.inv(self._ABDS)
        return self._ABDS

    def calculate_stresses(self, *args, **kwargs) -> ndarray:
        """
        Calculates material stresses for input internal forces or strains
        and returns it as a NumPy array.

        Either strains or stresses must be provided.

        If the points of evaluation are not explivitly specified with the parameter 'z',
        results are calculated at a default number of points per every layer.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None.
        stresses: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None
        z: Iterable[Number], Optional
            Points of evaluation. Default is None.
        rng: Iterable[Number], Optional
            The range in which 'z' is to be understood. Default is (-1, 1).
        squeeze: bool, Optional
            Whether to squeeze the reusulting array or not. Default is `True`.
        ppl: int, Optional
            Point per layer. Default is None.
        """
        return self._postprocess(*args, mode="stress", **kwargs)

    def utilization(self, *args, **kwargs) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.

        The implementation should be able to cover the case if the input 'strains' is a 2d array.
        In that case, the strain values are expected to run along the last axis, hence the i-th
        item would be accessed as `strains[i]` and it would return a tuple of numbers, one for
        every strain component involved in the formulation of the material law.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d array of strains such that the strains run along the last axis.
            The shape of this array determines the shape of the output in a straightforward
            manner.
        strains: numpy.ndarray, Optional
            1d or 2d array of strains such that the strains run along the last axis.
            The shape of this array determines the shape of the output in a straightforward
            manner.
        z: float or None, Optional
            The signed normal distance from the reference surface of the body.
            If `None`, values for all points are returned, grouped by layers (see later).
            Use it in combination with the 'rng' parameter.
        rng: Tuple[Number, Number], Optional
            An interval that puts the value of 'z' to perspective. Otherwise specified, the
            value for the parameter 'z' is expected in the range [-1, 1].

        Note
        ----
        The returned result treats layers as iterables even if the case of one single layer.
        This shows in the shapes of output arrays and you will quickly find the logic behind it
        with minimal experimentation.
        """
        return self._postprocess(*args, mode="u", **kwargs)

    def calculate_equivalent_stress(self, *args, **kwargs) -> ndarray:
        """
        Calculates equivalent material stresses for input internal forces or strains
        according to the built-in failure criteria and returns it as a NumPy array.

        Either strains or stresses must be provided.

        If the points of evaluation are not explivitly specified with the parameter 'z',
        results are calculated at a default number of points per every layer.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None.
        stresses: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None
        z: Iterable[Number], Optional
            Points of evaluation. Default is None.
        rng: Iterable[Number], Optional
            The range in which 'z' is to be understood. Default is (-1, 1).
        squeeze: bool, Optional
            Whether to squeeze the reusulting array or not. Default is `True`.
        ppl: int, Optional
            Point per layer. Default is None.
        """
        return self._postprocess(*args, mode="eq", **kwargs)

    def _postprocess(
        self,
        *,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
        z: Optional[Union[Number, Iterable[Number], None]] = None,
        rng: Optional[Tuple[Number, Number]] = (-1.0, 1.0),
        squeeze: Optional[bool] = True,
        ppl: Optional[Union[int, None]] = None,
        mode: Optional[str] = "stress",
        **kwargs
    ) -> Union[Number, Iterable[Number]]:
        """
        Calculates stresses, equivalent stresses or utilizations, according to the parameter
        'mode'. The different modes are:
            * 'stress' : the function returns stresses
            * 'eq': the function returns equivalent stresses
            * 'u': the function returns utilizations

        Either strains or stresses must be provided for the function.

        If the points of evaluation are not explivitly specified with the parameter 'z',
        results are calculated at a default number of points per every layer.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None.
        stresses: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None
        z: Iterable[Number], Optional
            Points of evaluation. Default is None.
        rng: Iterable[Number], Optional
            The range in which 'z' is to be understood. Default is (-1, 1).
        squeeze: bool, Optional
            Whether to squeeze the reusulting array or not. Default is `True`.
        ppl: int, Optional
            Point per layer. Default is None.
        mode: str, Optional
            This parameter controls the output.
            'stresses' : the function returns stresses
            'eq': the function returns equivalent stresses
            'u': the function returns utilizations
        """
        layers: Iterable[MindlinShellLayer] = self.layers

        return_stresses = mode == "stress"
        return_eq_stresses = mode == "eq"
        return_utilization = mode == "u"

        if z is None:
            num_layers = len(layers)
            num_data = strains.shape[0]
            num_point_per_layer = len(self.layer_class.__loc__) if ppl is None else ppl
            locations = np.linspace(-1.0, 1.0, num_point_per_layer)

            z, _layers = [], []
            for layer in layers:
                for loc in locations:
                    _layers.append(layer)
                    z.append(layer._loc_to_z(loc))

            result = self._postprocess_standard_form(
                strains=strains,
                stresses=stresses,
                z=z,
                rng=rng,
                squeeze=False,
                mode=mode,
                layers=_layers,
                **kwargs
            )

            if len(result.shape) == 2:
                assert return_eq_stresses or return_utilization
                result = result.reshape(num_data, num_layers, num_point_per_layer)
            else:
                assert return_stresses
                num_X = result.shape[-1]
                result = result.reshape(
                    num_data, num_layers, num_point_per_layer, num_X
                )

            return np.squeeze(result) if squeeze else result
        else:
            return self._postprocess_standard_form(
                strains=strains,
                stresses=stresses,
                z=z,
                rng=rng,
                squeeze=squeeze,
                mode=mode,
                **kwargs
            )

    def _postprocess_standard_form(
        self,
        *,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
        z: Optional[Union[Number, Iterable[Number], None]] = None,
        rng: Optional[Tuple[Number, Number]] = (-1.0, 1.0),
        squeeze: Optional[bool] = True,
        mode: Optional[str] = "stress",
        layers: Optional[Union[Iterable[MindlinShellLayer], None]] = None,
        **kwargs
    ) -> Union[Number, Iterable[Number]]:
        """
        Calculates stresses, equivalent stresses or utilizations, according to the parameter
        'mode'. The different modes are:
            * 'stresses' : the function returns stresses
            * 'eq': the function returns equivalent stresses
            * 'u': the function returns utilizations

        Either strains or stresses must be provided for the function.

        If the points of evaluation are not explivitly specified with the parameter 'z',
        results are calculated at a default number of points per every layer.

        Parameters
        ----------
        strains: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None.
        stresses: numpy.ndarray, Optional
            1d or 2d NumPy array. Default is None
        z: Iterable[Number], Optional
            Points of evaluation. Default is None.
        rng: Iterable[Number], Optional
            The range in which 'z' is to be understood. Default is (-1, 1).
        squeeze: bool, Optional
            Whether to squeeze the reusulting array or not. Default is `True`.
        ppl: int, Optional
            Point per layer. Default is None.
        layers: Iterable[MindlinShellLayer], Optional
            If layers are explicitly provided (for instance as leftover data from a previous
            calculation step), the number of layers must match the number of points provided
            by the parameter 'z'.
        mode: str, Optional
            This parameter controls the output.
            'stresses' : the function returns stresses
            'eq': the function returns equivalent stresses
            'u': the function returns utilizations
        """

        if strains is None:
            assert isinstance(stresses, ndarray)
            strains = (self.SDBA @ stresses.T).T

        strains = atleastnd(strains, 2, front=True)

        if stresses is None:
            stresses = (self.ABDS @ strains.T).T

        return_stresses = mode == "stress"
        return_eq_stresses = mode == "eq"
        return_utilization = mode == "u"

        assert any([return_stresses, return_eq_stresses, return_utilization])

        # mapping input points
        z = atleast1d(z)
        bounds = self.bounds
        z = to_range_1d(z, source=rng, target=(bounds[0, 0], bounds[-1, -1]))

        # mapping points to layers
        if layers is None:
            layers: Iterable[MindlinShellLayer] = self.find_layers(z, rng)

        num_z = len(layers)
        num_data = strains.shape[0]

        assert all([layer is not None for layer in layers])

        if return_stresses:
            result = np.zeros((num_data, num_z, 5), dtype=float)
        else:
            result = np.zeros((num_data, num_z), dtype=float)

        for iz, layer in enumerate(layers):
            C = layer.material_elastic_stiffness_matrix()
            material_stresses = np.zeros((num_data, 5), dtype=float)

            material_stresses[:, :3] = (
                C[:3, :3] @ (strains[:, :3] + z[iz] * strains[:, 3:6]).T
            ).T

            sfx, sfy = z_to_shear_factors(z[iz], layer._sfx, layer._sfy)
            material_stresses[:, 3] = sfx * stresses[:, -2]
            material_stresses[:, 4] = sfy * stresses[:, -1]

            if return_stresses:
                result[:, iz, :] = material_stresses
            elif return_eq_stresses:
                result[:, iz] = HMH_S_multi(material_stresses)
            elif return_utilization:
                result[:, iz] = (
                    HMH_S_multi(material_stresses) / layer.material.yield_strength
                )

        return np.squeeze(result) if squeeze else result
