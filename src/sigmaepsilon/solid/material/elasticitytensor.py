from typing import Union, Optional, Iterable, Tuple
from numbers import Number

from numpy import ndarray
import numpy as np

from sigmaepsilon.math.linalg import Tensor4
from sigmaepsilon.math.linalg.exceptions import TensorShapeMismatchError

from .utils.imap import _map_3x3x3x3_to_6x6, _map_6x6_to_3x3x3x3
from .utils.utils import _has_elastic_params, elastic_stiffness_matrix
from .utils.hmh import HMH_3d, HMH_3d_multi, HMH_3d_v

__all__ = ["ElasticityTensor"]


class ElasticityTensor(Tensor4):
    """
    A class to represent the 4th order stiffness tensor.

    Parameters
    ----------
    yield_strength: Number, Optional
        The maximum stress the material can sustain without experiencing
        relevant losses in performance. Default is `np.Infinity`.
    """

    def __init__(self, *args, yield_strength: Optional[Number] = np.Infinity, **kwargs):
        self._input_params = None
        if len(args) > 0 and isinstance(args[0], dict):
            if _has_elastic_params(args[0]):
                self._input_params = args[0]
                args = (elastic_stiffness_matrix(args[0]),)
        elif _has_elastic_params(**kwargs):
            self._input_params = kwargs
            args = (elastic_stiffness_matrix(**kwargs),)

        self._yield_strength = yield_strength

        if len(args) > 0 and isinstance(args[0], ndarray):
            arr = args[0]
            shape = arr.shape

            if shape[-1] == 6:
                if len(shape) >= 3:
                    is_bulk = kwargs.get("bulk", True)
                    if not is_bulk:
                        raise ValueError("Incorrect input!")
                    kwargs["bulk"] = is_bulk
                else:
                    if not len(shape) == 2:
                        raise TensorShapeMismatchError("Invalid shape!")
                    is_bulk = kwargs.get("bulk", False)
                    if is_bulk:
                        raise ValueError("Incorrect input!")

                arr = _map_6x6_to_3x3x3x3(arr)

            elif shape[-1] == 3:
                if len(shape) >= 5:
                    is_bulk = kwargs.get("bulk", True)
                    if not is_bulk:
                        raise ValueError("Incorrect input!")
                    kwargs["bulk"] = is_bulk
                else:
                    if not len(shape) == 4:
                        raise TensorShapeMismatchError("Invalid shape!")
                    is_bulk = kwargs.get("bulk", False)
                    if is_bulk:
                        raise ValueError("Incorrect input!")
            else:
                raise TensorShapeMismatchError("Invalid shape!")

            super().__init__(arr, *args[1:], **kwargs)
        else:
            super().__init__(*args, **kwargs)

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        if bulk:
            return (len(arr.shape) >= 5 and arr.shape[-4:] == (3, 3, 3, 3)) or (
                len(arr.shape) >= 3 and arr.shape[-2:] == (6, 6)
            )
        else:
            return (len(arr.shape) == 4 and arr.shape[-4:] == (3, 3, 3, 3)) or (
                len(arr.shape) == 2 and arr.shape[-2:] == (6, 6)
            )

    @property
    def yield_strength(self) -> Union[Number, None]:
        """
        Returns a number, or None if there is no yield stress defined for the
        material.
        """
        return self._yield_strength

    @yield_strength.setter
    def yield_strength(self, value: Union[Number, None]) -> None:
        """
        Sets the yield strength.
        """
        self._yield_strength = np.Infinity if value is None else value
        if isinstance(self._input_params, dict):
            self._input_params["yield_strength"] = self._yield_strength

    @property
    def material_parameters(self) -> Union[dict, ndarray]:
        """
        Returns a dictionary of material parameters or a NumPy array if it was created.
        directly using a NumPy array. If a result is a dictionary, it can be used to
        instantiate a clone of this material (using the same material frame).
        """
        if not self._input_params:
            return self.array
        else:
            return self._input_params

    def contracted_components(self, *args, **kwargs) -> ndarray:
        """
        Returns the 2d matrix representation of the tensor.
        """
        if (len(args) + len(kwargs)) > 0:
            arr = self.show(*args, **kwargs)
        else:
            arr = self.array
        return _map_3x3x3x3_to_6x6(arr)

    def transpose(self, *_, **__) -> "ElasticityTensor":
        """
        Returns the instance itself without modification regardless
        of the parameters (since the object is symmetric).
        """
        return self

    def elastic_stiffness_matrix(self, *_, **__) -> ndarray:
        """
        This implementation is here to comply with protocols that
        enable the instance to be cosidered as a valid material rule.
        """
        return self.contracted_components()

    def utilization(
        self,
        *args,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
    ) -> Union[Number, Iterable[Number]]:
        """
        A function that returns a positive number. If the value is 1.0, it means that the material
        is at peak performance and any further increase in the loads is very likely to lead to failure
        of the material.

        The strains or stresses are expected in the order

            s11, s22, s33, s23, s13, s12 = sxx, syy, szz, syz, sxz, sxy

        which is the classical unfolding of the 2nd-order Cauchy stiffness tensor.

        Parameters
        ----------
        strains: numpy.ndarray
            A 1d or 2d array of strain components. If it is a 2d array, it is expected that
            `strains[i]` has the meaning of the i-th selection of strain components, therefore
            the length of the second axis must be 6.

        Returns
        -------
        Number or Iterable[Number]
            A single utilization value as a float or several as a 1d array.
        """
        return (
            self.calculate_equivalent_stress(*args, strains=strains, stresses=stresses)
            / self.yield_strength
        )

    def calculate_equivalent_stress(
        self,
        *args,
        strains: Optional[Union[ndarray, None]] = None,
        stresses: Optional[Union[ndarray, None]] = None,
    ) -> ndarray:
        """
        A function that calculates the equivalent stress.

        The strains are expected in the order

            s11, s22, s33, s23, s13, s12 = sxx, syy, szz, syz, sxz, sxy

        which is the classical unfolding of the 2nd-order Cauchy stiffness tensor.

        Parameters
        ----------
        strains: numpy.ndarray
            A 1d or 2d array of strain components. If it is a 2d array, it is expected that
            `strains[i]` has the meaning of the i-th selection of strain components, therefore
            the length of the second axis must be 6.

        Returns
        -------
        Number or Iterable[Number]
            A single value as a float or several as a 1d array.
        """
        if strains is None and stresses is None:
            if len(args) == 1:
                if isinstance(args[0], ndarray):
                    return self.calculate_equivalent_stress(strains=args[0])
            elif len(args) == 6:
                return self.calculate_equivalent_stress(strains=args)

        if isinstance(strains, Tuple):
            if len(strains) == 6:
                stresses = self.contracted_components() @ np.stack(strains, axis=0)
                s11, s22, s33, s23, s13, s12 = stresses
                return HMH_3d_v(s11, s22, s33, s23, s13, s12)

            raise ValueError(
                (
                    "Invalid shape. If 'strains' is a tuple, it must be a 6-tuple "
                    "of 1d NumPy arrays of the same length."
                )
            )

        if not strains.shape[-1] == 6:
            raise ValueError("Invalid number of strain components.")

        if len(strains.shape) == 1:
            stresses = self.contracted_components() @ strains
            result = HMH_3d(stresses)
        else:
            stresses = (self.contracted_components() @ strains.T).T
            result = HMH_3d_multi(stresses)

        return result

    def calculate_stresses(self, strains: ndarray) -> ndarray:
        """
        A function that returns stresses for strains as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        if not strains.shape[-1] == 6:
            raise ValueError("Invalid number of strain components.")

        stresses = None

        if len(strains.shape) == 1:
            stresses = self.contracted_components() @ strains
        else:
            stresses = (self.contracted_components() @ strains.T).T

        return stresses

    def calculate_strains(self, stresses: ndarray) -> ndarray:
        """
        A function that returns strains for stresses as either an 1d or a 2d NumPy array,
        depending on the shape of the input array.
        """
        if not stresses.shape[-1] == 6:
            raise ValueError("Invalid number of strain components.")

        strains = None

        if len(stresses.shape) == 1:
            strains = np.linalg.inv(self.contracted_components()) @ stresses
        else:
            strains = (np.linalg.inv(self.contracted_components()) @ stresses.T).T

        return strains
