from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


# Raceinfo / RaceinfoPlayer offsets from SeekyCt/mkw-structures (PAL ROM).
class Raceinfo:
    OFFSET_PLAYERS = 0x0C
    OFFSET_STAGE = 0x28


class RaceinfoPlayer:
    OFFSET_POSITION = 0x20
    OFFSET_CURRENT_LAP_U16 = 0x24
    OFFSET_STATE_FLAGS_U32 = 0x34
    RACE_FINISH_FLAG_MASK = 0x20


# Raceinfo::sInstance DOL static pointer (PAL). Common tables use the same for RMCE01;
# override in config.yaml if mismatched on your DOL.
DEFAULT_RACEINFO_SINGLETON_BY_REGION: dict[str, int] = {
    "PAL": 0x809BD730,
    "RMCP01": 0x809BD730,
    "RMCE01": 0x809BD730,
    "RMCJ01": 0x809BD730,
    "RMCK01": 0x809BD730,
}


def raceinfo_singleton_for_region(region: str, override: int | None) -> int:
    if override is not None:
        return override
    key = region.strip().upper()
    return DEFAULT_RACEINFO_SINGLETON_BY_REGION.get(key, DEFAULT_RACEINFO_SINGLETON_BY_REGION["PAL"])


@dataclass(slots=True)
class MKWMemoryOverlay:
    place: int | None = None
    lap: int | None = None
    in_race: bool | None = None
    race_finished: bool | None = None


def try_import_dolphin_memory_engine() -> Any | None:
    try:
        import dolphin_memory_engine as dme
    except ImportError:
        return None
    return dme


def _guest_ptr_ok(ptr: int) -> bool:
    return 0x80000000 <= ptr < 0x81800000


class MKWDolphinMemoryReader:
    """
    Reads live Raceinfo / RaceinfoPlayer fields from Dolphin MEM1 via dolphin-memory-engine.
    """

    def __init__(
        self,
        raceinfo_singleton: int,
        player_slot: int = 0,
    ) -> None:
        self._singleton = raceinfo_singleton
        self._player_slot = max(0, min(11, player_slot))

    def read_overlay(self) -> MKWMemoryOverlay | None:
        dme = try_import_dolphin_memory_engine()
        if dme is None:
            return None

        if not dme.is_hooked():
            try:
                dme.hook()
            except RuntimeError:
                return None
        if not dme.is_hooked():
            return None

        try:
            rp = struct.unpack_from(
                ">I",
                dme.read_bytes(self._singleton, 4),
                0,
            )[0]
            if not _guest_ptr_ok(rp):
                return None

            players_pp = struct.unpack_from(
                ">I",
                dme.read_bytes(rp + Raceinfo.OFFSET_PLAYERS, 4),
                0,
            )[0]
            if not _guest_ptr_ok(players_pp):
                return None

            rplayer = struct.unpack_from(
                ">I",
                dme.read_bytes(players_pp + self._player_slot * 4, 4),
                0,
            )[0]
            if not _guest_ptr_ok(rplayer):
                return None

            place = struct.unpack_from(
                "B",
                dme.read_bytes(rplayer + RaceinfoPlayer.OFFSET_POSITION, 1),
                0,
            )[0]
            if place < 1 or place > 12:
                return None

            lap_raw = struct.unpack_from(
                ">H",
                dme.read_bytes(rplayer + RaceinfoPlayer.OFFSET_CURRENT_LAP_U16, 2),
                0,
            )[0]
            lap: int | None
            if 1 <= lap_raw <= 3:
                lap = lap_raw
            elif lap_raw == 0:
                lap = None
            else:
                lap = lap_raw if lap_raw <= 9 else None

            stage = struct.unpack_from(
                ">I",
                dme.read_bytes(rp + Raceinfo.OFFSET_STAGE, 4),
                0,
            )[0]

            flags = struct.unpack_from(
                ">I",
                dme.read_bytes(rplayer + RaceinfoPlayer.OFFSET_STATE_FLAGS_U32, 4),
                0,
            )[0]

            finishing = bool(flags & RaceinfoPlayer.RACE_FINISH_FLAG_MASK)
            in_race = stage == 2
            return MKWMemoryOverlay(
                place=place,
                lap=lap,
                in_race=in_race,
                race_finished=finishing,
            )
        except (RuntimeError, struct.error):
            return None
