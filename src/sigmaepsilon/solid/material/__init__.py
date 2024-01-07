from os.path import dirname, abspath
from importlib.metadata import metadata

from sigmaepsilon.core.config import namespace_package_name

from .frame import BernoulliFrameSection
from .surface import (
    MindlinShellSection,
    MembraneSection,
    MindlinPlateSection,
    KirchhoffPlateSection,
    KirchhoffShellSection,
)
from .stresstensor import CauchyStressTensor
from .straintensor import SmallStrainTensor
from .elasticitytensor import ElasticityTensor
from .failure import (
    HuberMisesHenckyFailureCriterion,
    HuberMisesHenckyFailureCriterion_SP,
    HuberMisesHenckyFailureCriterion_M,
    HoffmanFailureCriterion,
    HoffmanFailureCriterion_SP,
    HoffmanFailureCriterion_M,
)
from .linearelasticmaterial import LinearElasticMaterial
from .proto import MaterialLike, StiffnessLike, FailureLike

__all__ = [
    "BernoulliFrameSection",
    "MindlinShellSection",
    "KirchhoffPlateSection",
    "KirchhoffShellSection",
    "MembraneSection",
    "MindlinPlateSection",
    "CauchyStressTensor",
    "SmallStrainTensor",
    "ElasticityTensor",
    "HuberMisesHenckyFailureCriterion",
    "HuberMisesHenckyFailureCriterion_SP",
    "HuberMisesHenckyFailureCriterion_M",
    "HoffmanFailureCriterion",
    "HoffmanFailureCriterion_SP",
    "HoffmanFailureCriterion_M",
    "LinearElasticMaterial",
    "MaterialLike",
    "StiffnessLike",
    "FailureLike",
]

#__pkg_name__ = namespace_package_name(dirname(abspath(__file__)), 10)
__pkg_name__="sigmaepsilon.solid.material"
__pkg_metadata__ = metadata(__pkg_name__)
__version__ = __pkg_metadata__["version"]
__description__ = __pkg_metadata__["summary"]
del __pkg_metadata__