=============================================================================
**SigmaEpsilon.Solid.Material** - Classes and algorithms for solids in Python
=============================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   User Guide <user_guide>
   Gallery <examples_gallery>
   API Reference <api>
   Development <development>

**Version**: |version|

**Useful links**:
:doc:`getting_started` |
:doc:`user_guide` |
:doc:`examples_gallery` |
:ref:`API Reference` |
`Source Repository <https://github.com/sigma-epsilon/sigmaepsilon.solid.material>`_

.. _sigmaepsilon.solid.material: https://sigmaepsilon.solid.material.readthedocs.io/en/latest/
.. _Matplotlib: https://matplotlib.org/
.. _NumPy: https://numpy.org/doc/stable/index.html
.. _Numba: https://numba.pydata.org/
.. _SciPy: https://scipy.org/
.. _SymPy: https://www.sympy.org/en/index.html
.. _CuPy: https://cupy.dev/

The `sigmaepsilon.solid.material`_ is a set of tools to handle problems to related to solid mechanics,
namely those with questions of stiffness and strength of materials.

The implementations in the library all rely on fast and efficient algorithms provided by the goodies of
`NumPy`_, `SciPy`_ and the likes. Where necessary, computationally intensive parts of the code are written
using `Numba`_. Some evaluators also support calculating on the GPU (mostly on NVIDIA), for which we are 
using `CuPy`_ and `Numba`_. Symbolic calculations are done with `SymPy`_.

Highlights
==========

* Classes to handle linear elastic materials of all kinds.
* Elastic stiffness calculations for all kinds of models like Uflyand-Mindlin shells, 
  Kirchhoff-Love shells, Timoshenko-Ehrenfest and Euler-Bernoulli beams, 3d bodies, etc.
* Utilization calculations.
* Fitting of failure models to observed data.
* NumPy-compilant data classes to handle stiffness, strains and stresses.
* Fast and efficient code with GPU support.

Installation
============

You can install the project from PyPI with `pip`:

.. code-block:: shell

   $ pip install sigmaepsilon.solid.material

If want to execute on the GPU, you need to manually install the necessary requirements. 
Numba is a direct dependency, so even in this case you have to care about having the prover
version of the cuda toolkit installed. For this, you need to know the version of the cuda
compute engine, which depends on the version of GPU card you are having.


Contents
========

.. grid:: 2
    
    .. grid-item-card::
        :img-top: ../source/_static/index-images/getting_started.svg

        Getting Started
        ^^^^^^^^^^^^^^^

        The getting started guide contains a basic introduction to the main concepts 
        through simple examples.

        +++

        .. button-ref:: getting_started
            :expand:
            :color: secondary
            :click-parent:

            Get me started

    .. grid-item-card::
        :img-top: ../source/_static/index-images/user_guide.svg

        User Guide
        ^^^^^^^^^^

        The user guide provides a more detailed walkthrough of the library, touching 
        the key features with useful background information and explanation.

        +++

        .. button-ref:: user_guide
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in the library. The reference describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/gallery.svg

        Examples Gallery
        ^^^^^^^^^^^^^^^^

        A gallery of examples that illustrate uses cases that involve some
        kind of visualization.

        +++

        .. button-ref:: examples_gallery
            :expand:
            :color: secondary
            :click-parent:

            To the examples gallery
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



