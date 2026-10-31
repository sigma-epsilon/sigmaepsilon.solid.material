import unittest

import numpy as np
import xarray as xr

from sigmaepsilon.math.linalg import ReferenceFrame

from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material import (
    MembraneSection,
    MindlinShellSection,
    MindlinPlateSection,
    KirchhoffPlateSection,
    KirchhoffShellSection,
    LinearElasticMaterial,
    ElasticityTensor,
)
from sigmaepsilon.solid.material.surface.surface import SurfaceSection
from sigmaepsilon.solid.material.surface.mindlinshell import MindlinShellLayer
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix
from sigmaepsilon.solid.material.utils.mindlin import (
    _get_shell_material_stiffness_matrix,
)

E, NU, YIELD_STRENGTH, THICKNESS = 210000.0, 0.3, 355.0, 0.1


def in_plane_stiffness() -> np.ndarray:
    """
    The in-plane block of the material stiffness matrix the layers use.
    """
    hooke = elastic_stiffness_matrix(E=E, NU=NU)
    return _get_shell_material_stiffness_matrix(hooke)[:3, :3]


def homogeneous_section(cls, num_layers: int = 1):
    material = cls.Material(E=E, nu=NU, yield_strength=YIELD_STRENGTH)
    layers = [
        cls.Layer(material=material, thickness=THICKNESS / num_layers)
        for _ in range(num_layers)
    ]
    section = cls(layers=layers)
    section.elastic_stiffness_matrix()
    return section


class TestMembranePostprocessing(SolidMaterialTestCase):
    def setUp(self):
        self.section = homogeneous_section(MembraneSection)
        self.strains = np.random.rand(4, 3) / 1000

    def test_stiffness(self):
        ABDS = self.section.elastic_stiffness_matrix()
        self.assertTrue(np.allclose(ABDS, THICKNESS * in_plane_stiffness()))
        self.assertTrue(np.allclose(self.section.SDBA @ ABDS, np.eye(3)))

    def test_stresses(self):
        stresses = self.section.calculate_stresses(strains=self.strains, z=[-1, 0, 1])
        self.assertIsInstance(stresses, xr.DataArray)
        self.assertEqual(stresses.dims, ("index", "point", "component"))
        expected = (in_plane_stiffness() @ self.strains.T).T
        for i in range(3):
            self.assertTrue(np.allclose(stresses.values[:, i, :], expected))

        # the same from internal forces
        forces = (self.section.ABDS @ self.strains.T).T
        stresses = self.section.calculate_stresses(stresses=forces, z=[0.0])
        self.assertTrue(np.allclose(stresses.values, expected))

    def test_stresses_per_layer(self):
        section = homogeneous_section(MembraneSection, num_layers=2)
        stresses = section.calculate_stresses(strains=self.strains)
        self.assertEqual(stresses.dims, ("index", "layer", "point", "component"))
        self.assertEqual(stresses.shape, (4, 2, 3, 3))

    def test_coordinates(self):
        _, coords = self.section.calculate_stresses(
            strains=self.strains, z=[-1, 1], return_coords=True
        )
        self.assertTrue(np.allclose(coords, [[-THICKNESS / 2, THICKNESS / 2]] * 4))

        points = np.random.rand(4, 3)
        _, coords = self.section.calculate_stresses(
            strains=self.strains, z=[-1, 1], return_coords=True, coords=points
        )
        self.assertEqual(coords.shape, (4, 2, 3))
        self.assertTrue(np.allclose(coords[:, 0, :2], points[:, :2]))
        self.assertTrue(np.allclose(coords[:, 1, 2], THICKNESS / 2))

    def test_principal_stresses(self):
        strains = np.array([[1.0, -0.5, 0.3]]) / 1000
        sxx, syy, sxy = in_plane_stiffness() @ strains[0]
        center = (sxx + syy) / 2
        radius = np.sqrt(((sxx - syy) / 2) ** 2 + sxy**2)

        for z in ([0.0], None):
            principal = self.section.principal_material_stresses(
                strains=strains, z=z, squeeze=False
            )
            self.assertTrue(np.allclose(principal.sel(component="s1"), center + radius))
            self.assertTrue(np.allclose(principal.sel(component="s2"), center - radius))

        values, vectors = self.section.eig(strains=strains, z=[-1, 0, 1])
        self.assertTrue(np.allclose(values, [center - radius, center + radius]))
        self.assertEqual(vectors.shape, (3, 2, 2))
        self.assertFailsProperly(NotImplementedError, self.section.eig, strains=strains)

    def test_principal_stress_angles(self):
        angles = self.section.principal_material_stress_angles(
            strains=self.strains, z=[-1, 0, 1]
        )
        self.assertEqual(angles.dims, ("index", "point"))
        section = homogeneous_section(MembraneSection, num_layers=2)
        angles = section.principal_material_stress_angles(strains=self.strains)
        self.assertEqual(angles.dims, ("index", "layer", "point"))
        principal = section.principal_material_stresses(strains=self.strains)
        self.assertEqual(principal.dims, ("index", "layer", "point", "component"))

        # uniaxial stress in x has a principal direction of 0
        angles = self.section.principal_material_stress_angles(
            strains=np.array([[1.0, -NU, 0.0], [1.0, -NU, 0.0]]), z=[0.0], squeeze=False
        )
        self.assertTrue(np.allclose(angles, 0.0))

    def test_layer_shapes(self):
        layer = self.section.layers[0]
        self.assertEqual(layer.material_elastic_stiffness_matrix().shape, (5, 5))
        contracted = layer.material_elastic_stiffness_matrix("contracted")
        self.assertTrue(np.allclose(contracted, in_plane_stiffness()))
        self.assertEqual(layer.rotation_matrix("contracted").shape, (3, 3))
        self.assertFailsProperly(
            ValueError, layer.material_elastic_stiffness_matrix, "invalid"
        )
        self.assertFailsProperly(ValueError, layer.rotation_matrix, "invalid")
        self.assertFailsProperly(NotImplementedError, layer.rotation_matrix, None)


class TestMindlinShellPostprocessing(SolidMaterialTestCase):
    def setUp(self):
        self.section = homogeneous_section(MindlinShellSection)
        self.strains = np.random.rand(4, 8) / 1000

    def test_stresses(self):
        strains = self.strains
        stresses = self.section.calculate_stresses(strains=strains, z=[-1, 0, 1])
        self.assertEqual(stresses.dims, ("index", "point", "component"))

        # in-plane stresses vary linearly through the thickness
        C = in_plane_stiffness()
        for i, z in enumerate([-THICKNESS / 2, 0.0, THICKNESS / 2]):
            expected = (C @ (strains[:, :3] + z * strains[:, 3:6]).T).T
            self.assertTrue(np.allclose(stresses.values[:, i, :3], expected))

        # parabolic shear stress distribution, with a peak of 1.5 * Q / t
        forces = (self.section.ABDS @ strains.T).T
        sxz = stresses.sel(component="SXZ").values
        syz = stresses.sel(component="SYZ").values
        self.assertTrue(np.allclose(sxz[:, [0, 2]], 0.0))
        self.assertTrue(np.allclose(syz[:, [0, 2]], 0.0))
        self.assertTrue(np.allclose(sxz[:, 1], 1.5 * forces[:, 6] / THICKNESS))
        self.assertTrue(np.allclose(syz[:, 1], 1.5 * forces[:, 7] / THICKNESS))

        # the same from internal forces
        _stresses = self.section.calculate_stresses(stresses=forces, z=[-1, 0, 1])
        self.assertTrue(np.allclose(_stresses.values, stresses.values))

    def test_stresses_per_layer(self):
        section = homogeneous_section(MindlinShellSection, num_layers=2)
        stresses = section.calculate_stresses(strains=self.strains)
        self.assertEqual(stresses.dims, ("index", "layer", "point", "component"))
        self.assertEqual(stresses.shape, (4, 2, 3, 5))
        stresses = section.calculate_stresses(strains=self.strains, ppl=5)
        self.assertEqual(stresses.shape, (4, 2, 5, 5))

    def test_utilization(self):
        section = homogeneous_section(MindlinShellSection, num_layers=2)
        stresses = section.calculate_stresses(strains=self.strains, z=[-1, 0, 1])
        failure_model = section.layers[0].material.failure_model
        expected = failure_model.utilization(
            stresses=stresses.values.reshape(-1, 5)
        ).reshape(4, 3)
        utilization = section.utilization(strains=self.strains, z=[-1, 0, 1])
        self.assertTrue(np.allclose(utilization, expected))
        self.assertEqual(section.utilization(strains=self.strains).shape, (4, 2, 3))

    def test_layer(self):
        layer: MindlinShellLayer = self.section.layers[0]
        self.assertEqual(layer.shear_factors.shape, (3, 2))

        # the approximator reproduces quadratic data
        func = lambda z: 1.0 + 2.0 * z + 3.0 * z**2
        approximator = layer.approximator([func(z) for z in layer._zi])
        self.assertAlmostEqual(approximator(0.01), func(0.01))

        # shear factors are interpolated quadratically
        layer._sfx = layer._sfy = None
        sfx_mid, _ = layer._loc_to_shear_factors(0.0)
        self.assertAlmostEqual(sfx_mid, layer.shear_factors[1, 0])
        sfx_top, _ = layer._loc_to_shear_factors(1.0)
        self.assertAlmostEqual(sfx_top, 0.0)

        forces = (self.section.ABDS @ self.strains.T).T
        out = layer.stresses(strains=self.strains, stresses=forces, z=0.0)
        self.assertEqual(out.shape, (4, 5))


class TestPlatePostprocessing(SolidMaterialTestCase):
    def test_mindlin_plate_layer(self):
        section = homogeneous_section(MindlinPlateSection)
        curvatures = np.random.rand(4, 5) / 1000
        forces = (section.ABDS @ curvatures.T).T
        out = section.layers[0].stresses(
            strains=curvatures, stresses=forces, z=THICKNESS / 2
        )
        expected = (in_plane_stiffness() @ (THICKNESS / 2 * curvatures[:, :3]).T).T
        self.assertTrue(np.allclose(out[:, :3], expected))

    def test_kirchhoff_plate(self):
        section = homogeneous_section(KirchhoffPlateSection)
        curvatures = np.random.rand(4, 3) / 1000
        shear_forces = np.random.rand(4, 2)
        stresses = section.calculate_stresses(
            strains=curvatures, shear_forces=shear_forces, z=[-1, 0, 1]
        )
        expected = (in_plane_stiffness() @ (THICKNESS / 2 * curvatures).T).T
        self.assertTrue(np.allclose(stresses.values[:, 2, :3], expected))
        self.assertTrue(np.allclose(stresses.values[:, 1, :3], 0.0))
        self.assertTrue(
            np.allclose(stresses.values[:, 1, 3:], 1.5 * shear_forces / THICKNESS)
        )

        # the same from internal forces
        forces = np.zeros((4, 5))
        forces[:, :3] = (section.ABDS @ curvatures.T).T
        forces[:, 3:] = shear_forces
        _stresses = section.calculate_stresses(stresses=forces, z=[-1, 0, 1])
        self.assertTrue(np.allclose(_stresses.values, stresses.values))

        self.assertFailsProperly(Exception, setattr, section, "eccentricity", 1.0)

    def test_kirchhoff_shell(self):
        section = homogeneous_section(KirchhoffShellSection)
        strains = np.random.rand(4, 6) / 1000
        shear_forces = np.random.rand(4, 2)
        stresses = section.calculate_stresses(
            strains=strains, shear_forces=shear_forces, z=[-1, 0, 1]
        )
        C = in_plane_stiffness()
        expected = (C @ (strains[:, :3] + THICKNESS / 2 * strains[:, 3:]).T).T
        self.assertTrue(np.allclose(stresses.values[:, 2, :3], expected))
        self.assertTrue(
            np.allclose(stresses.values[:, 1, 3:], 1.5 * shear_forces / THICKNESS)
        )

        forces = np.zeros((4, 8))
        forces[:, :6] = (section.ABDS @ strains.T).T
        forces[:, 6:] = shear_forces
        _stresses = section.calculate_stresses(stresses=forces, z=[-1, 0, 1])
        self.assertTrue(np.allclose(_stresses.values, stresses.values))


class TestSurfaceBase(SolidMaterialTestCase):
    def setUp(self):
        self.hooke = elastic_stiffness_matrix(E=E, NU=NU)

    def test_material_types(self):
        stiffness = ElasticityTensor(self.hooke, frame=ReferenceFrame(dim=3))
        materials = [
            self.hooke,
            stiffness,
            LinearElasticMaterial(stiffness=self.hooke),
            LinearElasticMaterial(stiffness=stiffness),
        ]
        for material in materials:
            layer = MindlinShellSection.Layer(material=material, thickness=THICKNESS)
            self.assertTrue(np.allclose(layer.material_stiffness_matrix, self.hooke))

        layer = MindlinShellSection.Layer(material=self.hooke)
        layer.material_stiffness_matrix = 2 * self.hooke
        self.assertTrue(np.allclose(layer.material_stiffness_matrix, 2 * self.hooke))
        layer.material = None
        self.assertIsNone(layer.material)

    def test_layer_setters(self):
        layer = MindlinShellSection.Layer(material=self.hooke)
        layer.angle = 30.0
        self.assertEqual(layer.angle, 30.0)
        layer.thickness = 0.2
        self.assertEqual(layer.thickness, 0.2)
        self.assertFailsProperly(TypeError, setattr, layer, "angle", "a")
        self.assertFailsProperly(TypeError, setattr, layer, "thickness", "a")

    def test_section_properties(self):
        layers = [
            MindlinShellSection.Layer(material=self.hooke, thickness=0.1, angle=0.0),
            MindlinShellSection.Layer(material=self.hooke, thickness=0.2, angle=45.0),
        ]
        layout = {"a": layers[0], "b": layers[1]}
        section = MindlinShellSection(layers=layers, layout=layout)
        section.elastic_stiffness_matrix()
        self.assertIs(section.layout, layout)
        self.assertTrue(np.allclose(section.angles, [0.0, 45.0]))
        self.assertTrue(np.allclose(section.SDBA @ section.ABDS, np.eye(8)))
        self.assertTrue(np.allclose(section.bounds[[0, -1], [0, -1]], [-0.15, 0.15]))
        self.assertIs(section.find_layer(-0.1), layers[0])
        self.assertIs(section.find_layer(0.1), layers[1])
        self.assertIsNone(section.find_layer(1.0))

    def test_generic_stiffness_matrix(self):
        layers = [MindlinShellLayer(material=self.hooke, thickness=THICKNESS)]
        ABDS = SurfaceSection(layers=layers).elastic_stiffness_matrix()
        self.assertEqual(ABDS.shape, (8, 8))
        # without shear correction
        C = elastic_stiffness_matrix(E=E, NU=NU)
        self.assertAlmostEqual(ABDS[6, 6], C[-1, -1] * THICKNESS)
        self.assertTrue(np.allclose(ABDS[:3, :3], THICKNESS * in_plane_stiffness()))


if __name__ == "__main__":
    unittest.main()
