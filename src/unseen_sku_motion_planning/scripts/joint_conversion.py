#!/usr/bin/env python3

import math

def wrap_to_pi(angle: float) -> float:
    """
    Wraps angle to [-pi, pi].
    """
    return math.atan2(math.sin(angle), math.cos(angle))

def nearest_equivalent_angle(angle: float, reference: float) -> float:
    """
    Returns angle + 2*k*pi that is closest to reference.

    Example:
        angle     = 1.967
        reference = -4.316

    Returns approximately:
        -4.316
    """

    k = round((reference - angle) / (2.0 * math.pi))
    final_angle = angle + k * 2.0 * math.pi

    return final_angle

def arc_to_moma_position(
    arc_position: float,
    sign: float,
    offset: float,
    wrap: bool = True,
) -> float:
    """
    MOMA = sign * ARC + offset
    """

    moma_position = sign * arc_position + offset

    if wrap:
        moma_position = wrap_to_pi(moma_position)

    return moma_position

def moma_to_arc_position(
    moma_position: float,
    sign: float,
    offset: float,
    reference: float
) -> float:
    """
    ARC = (MOMA - offset) / sign

    Then choose the +/- 2*pi equivalent nearest to the current/previous ARC
    position.
    """
    raw_arc_position = (moma_position - offset) / sign

    return nearest_equivalent_angle(raw_arc_position, reference)

def arc_to_moma_derivative(value: float, sign: float) -> float:
    """
    Conversion for velocity / acceleration.

    Position offsets disappear when differentiated.
    """
    return sign * value

def moma_to_arc_derivative(value: float, sign: float) -> float:
    return value / sign
