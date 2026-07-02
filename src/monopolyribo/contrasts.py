# Imports ------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
# --------------------------------------------------


ContrastKind = Literal[
    'abundance',
    'fraction_vs_input',
    'omnibus_interaction',
    'redistribution'
]


@dataclass(frozen = True)
class FractionContrast:
    kind: ContrastKind
    case: str
    control: str
    numerator: str | None = None
    denominator: str | None = None
    fraction: str | None = None
    input_fraction: str | None = None

    def __post_init__(self) -> None:
        _validate_level('case', self.case)
        _validate_level('control', self.control)

        if self.case == self.control:
            raise ValueError('The case and control conditions must be different.')

        if self.kind == 'abundance':
            _validate_level('fraction', self.fraction)
            _require_missing('numerator', self.numerator)
            _require_missing('denominator', self.denominator)
            _require_missing('input_fraction', self.input_fraction)

        elif self.kind == 'redistribution':
            _validate_level('numerator', self.numerator)
            _validate_level('denominator', self.denominator)
            _require_missing('fraction', self.fraction)
            _require_missing('input_fraction', self.input_fraction)

            if self.numerator == self.denominator:
                raise ValueError('The numerator and denominator fractions must be different.')

        elif self.kind == 'fraction_vs_input':
            _validate_level('numerator', self.numerator)
            _validate_level('denominator', self.denominator)
            _validate_level('input_fraction', self.input_fraction)
            _require_missing('fraction', self.fraction)

            if self.denominator != self.input_fraction:
                raise ValueError('The denominator must match the input fraction.')

            if self.numerator == self.denominator:
                raise ValueError('The fraction and input fraction must be different.')

        elif self.kind == 'omnibus_interaction':
            _require_missing('numerator', self.numerator)
            _require_missing('denominator', self.denominator)
            _require_missing('fraction', self.fraction)
            _require_missing('input_fraction', self.input_fraction)

        else:
            raise ValueError(f'Unsupported contrast kind {self.kind!r}.')

    @classmethod
    def redistribution(
        cls,
        case: str,
        control: str,
        numerator: str,
        denominator: str
    ) -> FractionContrast:
        return cls(
            kind = 'redistribution',
            case = case,
            control = control,
            numerator = numerator,
            denominator = denominator
        )

    @classmethod
    def abundance(cls, case: str, control: str, fraction: str) -> FractionContrast:
        return cls(
            kind = 'abundance',
            case = case,
            control = control,
            fraction = fraction
        )

    @classmethod
    def fraction_vs_input(
        cls,
        case: str,
        control: str,
        fraction: str,
        input_fraction: str
    ) -> FractionContrast:
        return cls(
            kind = 'fraction_vs_input',
            case = case,
            control = control,
            numerator = fraction,
            denominator = input_fraction,
            input_fraction = input_fraction
        )

    @classmethod
    def omnibus_interaction(cls, case: str, control: str) -> FractionContrast:
        return cls(
            kind = 'omnibus_interaction',
            case = case,
            control = control
        )

    @property
    def label(self) -> str:
        condition_label = f'{self.case}_vs_{self.control}'

        if self.kind == 'abundance':
            return f'{condition_label}:{self.fraction}'

        if self.kind in {'redistribution', 'fraction_vs_input'}:
            return f'{condition_label}:{self.numerator}_vs_{self.denominator}'

        return f'{condition_label}:{self.kind}'


def _validate_level(name: str, value: str | None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'The {name} value must be a nonempty string.')


def _require_missing(name: str, value: str | None) -> None:
    if value is not None:
        raise ValueError(f'The {name} value is not valid for this contrast kind.')