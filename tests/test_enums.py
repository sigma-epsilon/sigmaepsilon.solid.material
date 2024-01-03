import unittest


from sigmaepsilon.solid.material.testing import SolidMaterialTestCase
from sigmaepsilon.solid.material.enums import MaterialModelType


class TestMaterialModelType(SolidMaterialTestCase):
    
    def test_membrane_stress_variables(self):
        self.assertEqual(MaterialModelType.MEMBRANE.number_of_stress_variables, 3)

    def test_plate_uflyand_mindlin_stress_variables(self):
        self.assertEqual(MaterialModelType.PLATE_UFLYAND_MINDLIN.number_of_stress_variables, 5)

    def test_plate_kirchhoff_love_stress_variables(self):
        self.assertEqual(MaterialModelType.PLATE_KIRCHHOFF_LOVE.number_of_stress_variables, 5)

    def test_shell_uflyand_mindlin_stress_variables(self):
        self.assertEqual(MaterialModelType.SHELL_UFLYAND_MINDLIN.number_of_stress_variables, 8)

    def test_shell_kirchhoff_love_stress_variables(self):
        self.assertEqual(MaterialModelType.SHELL_KIRCHHOFF_LOVE.number_of_stress_variables, 8)

    def test_default_stress_variables(self):
        self.assertEqual(MaterialModelType.DEFAULT.number_of_stress_variables, 6)
        
    def test_default_stress_variables(self):
        self.assertEqual(MaterialModelType.UNDEFINED.number_of_stress_variables, None)
        
    def test_membrane_stress_variables(self):
        self.assertEqual(MaterialModelType.MEMBRANE.number_of_material_stress_variables, 3)

    def test_plate_uflyand_mindlin_stress_variables(self):
        self.assertEqual(MaterialModelType.PLATE_UFLYAND_MINDLIN.number_of_material_stress_variables, 5)

    def test_plate_kirchhoff_love_stress_variables(self):
        self.assertEqual(MaterialModelType.PLATE_KIRCHHOFF_LOVE.number_of_material_stress_variables, 5)

    def test_shell_uflyand_mindlin_stress_variables(self):
        self.assertEqual(MaterialModelType.SHELL_UFLYAND_MINDLIN.number_of_material_stress_variables, 5)

    def test_shell_kirchhoff_love_stress_variables(self):
        self.assertEqual(MaterialModelType.SHELL_KIRCHHOFF_LOVE.number_of_material_stress_variables, 5)

    def test_default_stress_variables(self):
        self.assertEqual(MaterialModelType.DEFAULT.number_of_material_stress_variables, 6)
        
    def test_default_stress_variables(self):
        self.assertEqual(MaterialModelType.UNDEFINED.number_of_material_stress_variables, None)


if __name__ == "__main__":
    unittest.main()
