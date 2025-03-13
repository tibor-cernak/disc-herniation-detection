from enum import Enum


class MRIType(Enum):
    T1 = 0
    T2_TSE = 1
    T2_STIR = 2
    UNKNOWN = 3
