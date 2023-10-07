import unittest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.solid.core import MindlinShellSection as Section


class TestMindlinShellSection(SigmaEpsilonTestCase):
    def test_mindlin_shell_section(self):
        section = Section(
            layers=[
                Section.Layer(
                    material=Section.Material(E=2100000, nu=0.3), thickness=0.1
                )
            ]
        )
        section.elastic_stiffness_matrix()


if __name__ == "__main__":
    unittest.main()