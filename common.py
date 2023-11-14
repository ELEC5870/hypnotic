from dataclasses import dataclass


@dataclass(frozen=True)
class Area:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Entry:
    area: Area
    cost: float
    intra_mode: int
    isp_mode: int
    multi_ref_idx: int
    mip_flag: bool
    lfnst_idx: int
    mts_flag: int
    mpm: (int, int, int, int, int, int)
