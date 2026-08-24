"""Compatibility constructors used by NumPy random pickle payloads."""

from ._bit_generators import MT19937, PCG64, PCG64DXSM, Philox, SFC64, BitGenerator


_BIT_GENERATORS = {
    "MT19937": MT19937,
    "PCG64": PCG64,
    "PCG64DXSM": PCG64DXSM,
    "Philox": Philox,
    "SFC64": SFC64,
}


def __bit_generator_ctor(bit_generator="MT19937"):
    if isinstance(bit_generator, type):
        cls = bit_generator
    else:
        try:
            cls = _BIT_GENERATORS[bit_generator]
        except (KeyError, TypeError):
            raise ValueError(f"{bit_generator} is not a known BitGenerator module.") from None
    return cls()


def __generator_ctor(bit_generator_name="MT19937",
                     bit_generator_ctor=__bit_generator_ctor):
    from . import Generator
    if isinstance(bit_generator_name, BitGenerator):
        return Generator(bit_generator_name)
    return Generator(bit_generator_ctor(bit_generator_name))


def __randomstate_ctor(bit_generator_name="MT19937",
                       bit_generator_ctor=__bit_generator_ctor):
    from . import RandomState
    if isinstance(bit_generator_name, BitGenerator):
        return RandomState(bit_generator_name)
    return RandomState(bit_generator_ctor(bit_generator_name))
