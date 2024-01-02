from numpy import ndarray
from numba import vectorize

__cache = True


@vectorize("f8(" + ("f8, " * 15)[:-2] + ")", target="parallel", cache=__cache)
def Hoffman_failure_criterion_standard_form(
    s11, s22, s33, s23, s13, s12, C1, C2, C3, C4, C5, C6, C7, C8, C9
) -> ndarray:
    """
    Evaluates the Hoffman failure criterion in its canonical form.
    """
    return (
        C1 * (s22 - s33) ** 2
        + C2 * (s11 - s33) ** 2
        + C3 * (s22 - s11) ** 2
        + C4 * s11
        + C5 * s22
        + C6 * s33
        + C7 * s23
        + C8 * s13
        + C9 * s12
    )


@vectorize("f8(" + ("f8, " * 15)[:-2] + ")", target="parallel", cache=__cache)
def Hoffman_failure_criterion_principal_form(
    s11, s22, s33, s23, s13, s12, Xc, Xt, Yc, Yt, Zc, Zt, S23, S13, S12
) -> ndarray:
    """
    Evaluates the Hoffman failure criterion in its canonical form.
    """
    return (
        s33 / Zt
        + s33 / Zc
        - s11 * s22 / (Zc * Zt)
        + s11 * s33 / (Zc * Zt)
        + s22 * s33 / (Zc * Zt)
        - s33**2 / (Zc * Zt)
        + s22 / Yt
        + s22 / Yc
        + s11 * s22 / (Yc * Yt)
        - s11 * s33 / (Yc * Yt)
        - s22**2 / (Yc * Yt)
        + s22 * s33 / (Yc * Yt)
        + s11 / Xt
        + s11 / Xc
        - s11**2 / (Xc * Xt)
        + s11 * s22 / (Xc * Xt)
        + s11 * s33 / (Xc * Xt)
        - s22 * s33 / (Xc * Xt)
        + s23 / S23
        + s13 / S13
        + s12 / S12
    )
    

@vectorize("f8(" + ("f8, " * 14)[:-2] + ")", target="parallel", cache=__cache)
def Hoffman_failure_criterion_principal_form_PS(
    s11, s22, s23, s13, s12, Xc, Xt, Yc, Yt, Zc, Zt, S23, S13, S12
) -> ndarray:
    """
    Evaluates the Hoffman failure criterion for plates and shells in its principal form.
    Note that the parameters to feed the function are identical to those of the most
    general case, except that the stress s33 is missing.
    """
    return (
        - s11 * s22 / (Zc * Zt)
        + s22 / Yt
        + s22 / Yc
        + s11 * s22 / (Yc * Yt)
        - s22**2 / (Yc * Yt)
        + s11 / Xt
        + s11 / Xc
        - s11**2 / (Xc * Xt)
        + s11 * s22 / (Xc * Xt)
        + s23 / S23
        + s13 / S13
        + s12 / S12
    )
    

@vectorize("f8(" + ("f8, " * 10)[:-2] + ")", target="parallel", cache=__cache)
def Hoffman_failure_criterion_principal_form_M(
    s11, s22, s12, Xc, Xt, Yc, Yt, Zc, Zt, S12
) -> ndarray:
    """
    Evaluates the Hoffman failure criterion for membranes in its principal form.
    """
    return (
        - s11 * s22 / (Zc * Zt)
        + s22 / Yt
        + s22 / Yc
        + s11 * s22 / (Yc * Yt)
        - s22**2 / (Yc * Yt)
        + s11 / Xt
        + s11 / Xc
        - s11**2 / (Xc * Xt)
        + s11 * s22 / (Xc * Xt)
        + s12 / S12
    )
