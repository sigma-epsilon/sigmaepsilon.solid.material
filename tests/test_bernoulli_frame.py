import unittest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.solid.core import BernoulliFrameSection


class TestBernoulliFrameSection(SigmaEpsilonTestCase):
    def test_bernoulli_frame_section(self):
        section = BernoulliFrameSection("CHS", d=1.0, t=0.1, n=64)
        section.calculate_section_properties()


if __name__ == "__main__":
    unittest.main()
