import math

import torch

from torch import FloatTensor
from typing import Optional, NamedTuple

# Copied from https://github.com/Clybius/ComfyUI-Extra-Samplers


def _gamma(
    n: int,
) -> int:
    """
    https://en.wikipedia.org/wiki/Gamma_function
    for every positive integer n,
    Γ(n) = (n-1)!
    """
    return math.factorial(n - 1)


def _incomplete_gamma(s: int, x: float, gamma_s: Optional[int] = None) -> float:
    """
    https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
    if s is a positive integer,
    Γ(s, x) = (s-1)!*∑{k=0..s-1}(x^k/k!)
    """
    if gamma_s is None:
        gamma_s = _gamma(s)

    sum_: float = 0
    # {k=0..s-1} inclusive
    for k in range(s):
        numerator: float = x**k
        denom: int = math.factorial(k)
        quotient: float = numerator / denom
        sum_ += quotient
    incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
    return incomplete_gamma_


# by Katherine Crowson
def _phi_1(neg_h: FloatTensor):
    return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)


# by Katherine Crowson
def _phi_2(neg_h: FloatTensor):
    return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h**2, nan=0.5)


# by Katherine Crowson
def _phi_3(neg_h: FloatTensor):
    return torch.nan_to_num(
        (torch.expm1(neg_h) - neg_h - neg_h**2 / 2) / neg_h**3, nan=1 / 6
    )


def _phi(
    neg_h: float,
    j: int,
):
    """
    For j={1,2,3}: you could alternatively use Kat's phi_1, phi_2, phi_3 which perform fewer steps

    Lemma 1
    https://arxiv.org/abs/2308.02157
    ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

    https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
    = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
    https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
    = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
    = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
    = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
    = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
    = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

    requires j>0
    """
    assert j > 0
    gamma_: float = _gamma(j)
    incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)

    phi_: float = math.exp(neg_h) * neg_h**-j * (1 - incomp_gamma_ / gamma_)

    return phi_


class RESDECoeffsSecondOrder(NamedTuple):
    a2_1: float
    b1: float
    b2: float


def _de_second_order(
    h: float,
    c2: float,
    simple_phi_calc=False,
) -> RESDECoeffsSecondOrder:
    """
    Table 3
    https://arxiv.org/abs/2308.02157
    ϕi,j := ϕi,j(-h) = ϕi(-cj*h)
    a2_1 = c2ϕ1,2
         = c2ϕ1(-c2*h)
    b1 = ϕ1 - ϕ2/c2
    """
    if simple_phi_calc:
        # Kat computed simpler expressions for phi for cases j={1,2,3}
        a2_1: float = c2 * _phi_1(-c2 * h)
        phi1: float = _phi_1(-h)
        phi2: float = _phi_2(-h)
    else:
        # I computed general solution instead.
        # they're close, but there are slight differences. not sure which would be more prone to numerical error.
        a2_1: float = c2 * _phi(j=1, neg_h=-c2 * h)
        phi1: float = _phi(j=1, neg_h=-h)
        phi2: float = _phi(j=2, neg_h=-h)
    phi2_c2: float = phi2 / c2
    b1: float = phi1 - phi2_c2
    b2: float = phi2_c2
    return RESDECoeffsSecondOrder(
        a2_1=a2_1,
        b1=b1,
        b2=b2,
    )
