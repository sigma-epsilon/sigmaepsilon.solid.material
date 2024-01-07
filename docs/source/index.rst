=============================================================================================================
**SigmaEpsilon.Solid.Material** - Data Structures, Computation and Visualization for Complex Polygonal Meshes in Python
=============================================================================================================

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
`Source Repository <https://github.com/sigma-epsilon/sigmaepsilon.mesh>`_

.. _sigmaepsilon.solid.material: https://sigmaepsilon.solid.material.readthedocs.io/en/latest/
.. _Matplotlib: https://matplotlib.org/
.. _NumPy: https://numpy.org/doc/stable/index.html
.. _Numba: https://numba.pydata.org/
.. _SciPy: https://scipy.org/
.. _SymPy: https://www.sympy.org/en/index.html


The `sigmaepsilon.solid.material`_ library aims to provide the tools to build and analyse polygonal meshes 
with complex topologies. Meshes can be built like a dictionary, using arbitrarily nested layouts and 
then be translated to other formats including `VTK`_ and `PyVista`_. For plotting, there is also 
support for `K3D`_, `Matplotlib`_ and `Plotly`_.

The data model is built around `Awkward`_, which makes it possible to attach nested, variable-sized 
data to the points or the cells in a mesh, also providing interfaces to other popular libraries like 
`Pandas`_ or `PyArrow`_. Implementations are fast as they rely on the vector math capabilities of 
`NumPy`_, while other computationally sensitive calculations are JIT-compiled using `Numba`_.

Here and there we also use `SciPy`_, and `SymPy`_.


Highlights
==========

* Classes to handle linear elastic materials of all kinds.

Installation
============

You can install the project from PyPI with `pip`:

.. code-block:: shell

   $ pip install sigmaepsilon.solid.material


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



