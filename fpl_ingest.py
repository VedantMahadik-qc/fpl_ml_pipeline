"""
FPL ML Pipeline — Phase 1: Data Ingestion
=========================================

Two ingestion sources, one harmonised schema:

  LIVE       official FPL API (current season, refreshed every gameweek)
             bootstrap-static/       -> elements, teams, element_types
             element-summary/{id}/   -> history (per-gameweek rows)

  HISTORICAL vaastav/Fantasy-Premier-League GitHub archive (completed seasons)
             data/{season}/gws/merged_gw.csv
             data/{season}/teams.csv  (per-season opponent id -> short name)

Outputs
-------
  fpl_raw_gameweek_data.csv   live current-season rows only
  fpl_combined_history.csv    historical seasons + live, one tidy schema

Usage
-----
    python fpl_ingest.py                                  # live + historical
    python fpl_ingest.py --skip-historical                # live only
    python fpl_ingest.py --skip-live                      # archive only (fast)
    python fpl_ingest.py --seasons 2023-24 2024-25 2025-26
    python fpl_ingest.py --limit 25 --log-level DEBUG     # smoke test

Schema caveats (read these before modelling)
--------------------------------------------
* `player_id` and `opponent_team_id` are SEASON-SCOPED. Always join on
  (`season`, `player_id`), never on `player_id` alone. `player_name` is the
  practical cross-season key.
* `now_cost` is today's price and is only populated for the live season. Use
  `gw_cost` (price at that gameweek) as a feature — `now_cost` on historical
  rows would be lookahead.
* `xp` (FPL's own expected points) exists only in the archive; the live
  element-summary endpoint does not expose it. Useful as a benchmark, but it is
  NaN for the current season.
* `defensive_contribution` was introduced in 2025/26 — NaN for earlier seasons.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 v2 and v1 expose Retry at different paths
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
ELEMENT_SUMMARY_URL = f"{BASE_URL}/element-summary/{{player_id}}/"

VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
VAASTAV_MERGED_GW_URL = f"{VAASTAV_BASE}/{{season}}/gws/merged_gw.csv"
VAASTAV_TEAMS_URL = f"{VAASTAV_BASE}/{{season}}/teams.csv"

DEFAULT_SEASONS: List[str] = ["2023-24", "2024-25", "2025-26"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

LOGGER = logging.getLogger("fpl.ingest")

# Non-player rows: FPL added "Assistant Manager" (AM) in 2024/25, renamed MNG.
MANAGER_POSITIONS = {"AM", "MNG", "MAN"}

# Per-gameweek fields kept from the live `history` payload. Anything absent from
# a given season's response is created as NaN rather than raising, so the script
# survives the FPL API adding/removing stats between seasons.
HISTORY_FIELDS: List[str] = [
    "round",                          # gameweek number
    "fixture",
    "opponent_team",                  # season-scoped team id (mapped below)
    "was_home",
    "kickoff_time",
    "team_h_score",
    "team_a_score",
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",                 # FPL-provided underlying stats
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",         # added 2025/26; NaN on older seasons
    "value",                          # player price *at that gameweek* (tenths)
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
]

# vaastav merged_gw.csv -> our schema. Unlisted columns are carried through
# unchanged if they happen to match, otherwise dropped (and logged at DEBUG).
VAASTAV_RENAME: Dict[str, str] = {
    "name": "player_name",
    "element": "player_id",
    "xP": "xp",
    "opponent_team": "opponent_team_id",
}

# Columns coerced to numeric after the merge (the API/CSV return several as strings).
NUMERIC_COLS: List[str] = [
    "gameweek", "minutes", "total_points", "xp", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_saved",
    "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index", "starts",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution", "selected",
    "transfers_in", "transfers_out", "transfers_balance", "opponent_team_id",
    "team_h_score", "team_a_score", "now_cost", "gw_cost", "player_id", "team_id",
]

# Canonical output schema, shared by the live and historical paths.
OUTPUT_COLUMNS: List[str] = [
    "season", "player_id", "player_name", "web_name", "team", "team_id", "position",
    "now_cost", "gw_cost", "gameweek", "kickoff_time",
    "opponent_team", "opponent_team_id", "was_home",
    "team_h_score", "team_a_score", "minutes", "starts",
    "total_points", "xp", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution",
    "selected", "transfers_in", "transfers_out", "transfers_balance",
]

SORT_KEYS = ["season", "gameweek", "team", "player_name"]


@dataclass
class IngestConfig:
    """Runtime knobs for a single ingestion run."""

    output_path: Path = Path("fpl_raw_gameweek_data.csv")
    combined_path: Path = Path("fpl_combined_history.csv")
    seasons: Sequence[str] = tuple(DEFAULT_SEASONS)
    delay: float = 0.5                 # seconds between element-summary calls
    timeout: int = 20                  # per-request timeout (seconds)
    max_retries: int = 4
    limit: Optional[int] = None        # cap players processed (smoke tests)
    include_unavailable: bool = True   # keep injured/suspended players
    include_managers: bool = False     # keep AM/MNG rows from the archive
    skip_live: bool = False
    skip_historical: bool = False
    cache_dir: Optional[Path] = Path(".fpl_cache")
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

class FPLClient:
    """Retrying HTTP client for both the FPL API and the GitHub archive."""

    def __init__(self, config: IngestConfig) -> None:
        self.config = config
        self.session = self._build_session(config.max_retries)
        if config.cache_dir is not None:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
        retry = Retry(
            total=max_retries,
            backoff_factor=1.0,                       # 1s, 2s, 4s, 8s ...
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    # -- caching -------------------------------------------------------------
    def _cache_path(self, key: str, suffix: str) -> Optional[Path]:
        if self.config.cache_dir is None:
            return None
        return self.config.cache_dir / f"{key}{suffix}"

    def _write_cache_text(self, key: str, suffix: str, text: str) -> None:
        path = self._cache_path(key, suffix)
        if path is None:
            return
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            LOGGER.debug("Could not cache %s (%s)", key, exc)

    def _read_cache_text(self, key: str, suffix: str) -> Optional[str]:
        path = self._cache_path(key, suffix)
        if path is None or not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            LOGGER.debug("Cache read failed for %s (%s)", key, exc)
            return None

    # -- requests ------------------------------------------------------------
    def _get(self, url: str) -> Optional[requests.Response]:
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as exc:
            LOGGER.warning("HTTP error for %s: %s", url, exc)
        except requests.exceptions.Timeout:
            LOGGER.warning("Timed out after %ss: %s", self.config.timeout, url)
        except requests.exceptions.RequestException as exc:
            LOGGER.warning("Request failed for %s: %s", url, exc)
        return None

    def get_json(self, url: str, cache_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """GET a JSON document. Returns None on any unrecoverable failure."""
        if cache_key:
            cached = self._read_cache_text(cache_key, ".json")
            if cached is not None:
                try:
                    LOGGER.debug("Cache hit: %s", cache_key)
                    return json.loads(cached)
                except json.JSONDecodeError:
                    LOGGER.debug("Corrupt cache for %s, refetching", cache_key)

        response = self._get(url)
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            LOGGER.warning("Non-JSON response from %s: %s", url, exc)
            return None

        if cache_key:
            self._write_cache_text(cache_key, ".json", response.text)
        return payload

    def get_text(self, url: str, cache_key: Optional[str] = None) -> Optional[str]:
        """GET a text document (used for the archive CSVs)."""
        if cache_key:
            cached = self._read_cache_text(cache_key, ".csv")
            if cached is not None:
                LOGGER.debug("Cache hit: %s", cache_key)
                return cached

        response = self._get(url)
        if response is None:
            return None
        response.encoding = response.encoding or "utf-8"
        if cache_key:
            self._write_cache_text(cache_key, ".csv", response.text)
        return response.text

    # -- FPL endpoints -------------------------------------------------------
    def fetch_bootstrap(self) -> Dict[str, Any]:
        """Fetch bootstrap-static. Raises if unavailable — nothing works without it."""
        LOGGER.info("Fetching bootstrap-static ...")
        payload = self.get_json(BOOTSTRAP_URL, cache_key="bootstrap_static")
        if not payload:
            raise RuntimeError("Could not fetch bootstrap-static; aborting live ingestion.")
        for key in ("elements", "teams", "element_types"):
            if key not in payload:
                raise RuntimeError(f"bootstrap-static payload missing '{key}' key.")
        LOGGER.info(
            "bootstrap-static OK: %d players, %d teams, %d positions",
            len(payload["elements"]), len(payload["teams"]), len(payload["element_types"]),
        )
        return payload

    def fetch_player_history(self, player_id: int) -> List[Dict[str, Any]]:
        """Fetch one player's current-season gameweek history. [] if unavailable."""
        url = ELEMENT_SUMMARY_URL.format(player_id=player_id)
        payload = self.get_json(url, cache_key=f"element_summary_{player_id}")
        if not payload:
            return []
        history = payload.get("history")
        if not isinstance(history, list):
            LOGGER.warning("Player %s: unexpected 'history' payload", player_id)
            return []
        return history


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def normalise_position(value: Any) -> str:
    """FPL API says 'GKP'; the vaastav archive says 'GK'. Standardise on 'GKP'."""
    if not isinstance(value, str):
        return "UNK"
    token = value.strip().upper()
    return {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}.get(
        token, token or "UNK"
    )


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """The API/CSV return xG/ICT stats as strings — make them numeric for modelling."""
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def coerce_bool(series: pd.Series) -> pd.Series:
    """'True'/'true'/1/True -> True, everything falsy -> False, blanks -> NA."""
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    )
    return mapped.astype("boolean")


def conform_to_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every OUTPUT_COLUMN exists, in order, dropping stray columns."""
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    dropped = [c for c in frame.columns if c not in OUTPUT_COLUMNS]
    if dropped:
        LOGGER.debug("Dropping non-schema columns: %s", ", ".join(sorted(dropped)))
    return frame[OUTPUT_COLUMNS].copy()   # .copy() -> avoids SettingWithCopyWarning downstream


def finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Shared post-processing: dtypes, timestamps, ordering."""
    if frame.empty:
        return frame
    frame = coerce_numeric(frame.copy(), NUMERIC_COLS)
    if "was_home" in frame.columns:
        frame["was_home"] = coerce_bool(frame["was_home"])
    if "kickoff_time" in frame.columns:
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)
    if "position" in frame.columns:
        frame["position"] = frame["position"].map(normalise_position)
    return frame.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Live ingestion — transform layer (pure functions, no network)
# --------------------------------------------------------------------------- #

# Long club names mapped to the short form the archive has always used. The
# live API switched to "Ipswich Town", "Coventry City" and friends in 2026-27;
# team ratings carry across seasons by club NAME, so without this one club
# becomes two the moment its name changes.
CLUB_NAME_ALIASES: Dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Spurs",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def canonical_club(name: Any) -> Any:
    """One name per club across every season and source."""
    if not isinstance(name, str):
        return name
    return CLUB_NAME_ALIASES.get(name.strip(), name.strip())


def build_team_lookup(teams: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """team_id -> {'name', 'short_name'}. Season-scoped: ids are reused each year."""
    return {
        int(t["id"]): {"name": canonical_club(t.get("name", "Unknown")),
                       "short_name": t.get("short_name", "UNK")}
        for t in teams
    }


def build_position_lookup(element_types: Iterable[Dict[str, Any]]) -> Dict[int, str]:
    """element_type id -> 'GKP' / 'DEF' / 'MID' / 'FWD'"""
    return {
        int(et["id"]): normalise_position(
            et.get("singular_name_short") or et.get("singular_name", "UNK")
        )
        for et in element_types
    }


def derive_season(bootstrap: Dict[str, Any]) -> str:
    """Infer '2026-27' from the earliest event deadline. Falls back to 'current'."""
    events = bootstrap.get("events") or []
    deadlines = [e.get("deadline_time") for e in events if e.get("deadline_time")]
    if not deadlines:
        return "current"
    try:
        start_year = min(pd.to_datetime(deadlines, utc=True, errors="coerce")).year
    except (ValueError, TypeError):
        return "current"
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def build_player_frame(
    elements: Iterable[Dict[str, Any]],
    team_lookup: Dict[int, Dict[str, str]],
    position_lookup: Dict[int, str],
) -> pd.DataFrame:
    """Static, one-row-per-player metadata frame."""
    rows = []
    for el in elements:
        team = team_lookup.get(el.get("team"), {})
        rows.append(
            {
                "player_id": el["id"],
                "player_name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
                "web_name": el.get("web_name", ""),
                "team": team.get("name", "Unknown"),
                # Season-scoped, like every other FPL id — but it makes the
                # opponent join in Phase 2 exact instead of name-matched.
                "team_id": el.get("team"),
                "position": position_lookup.get(el.get("element_type"), "UNK"),
                # now_cost is in tenths of a million (55 -> £5.5m)
                "now_cost": el.get("now_cost", 0) / 10.0,
                "status": el.get("status", "a"),
            }
        )
    return pd.DataFrame(rows)


def select_active_players(players: pd.DataFrame, include_unavailable: bool = True) -> pd.DataFrame:
    """
    Filter to players worth requesting.

    FPL `status` codes: a=available, d=doubtful, i=injured, s=suspended,
    u=unavailable, n=not in squad. Injured/doubtful players still have valuable
    history, so they are kept by default; only 'u'/'n' are dropped, and only
    when include_unavailable is False.
    """
    if include_unavailable:
        return players
    return players[~players["status"].isin(["u", "n"])].reset_index(drop=True)


def normalise_history(history: List[Dict[str, Any]], player_id: int) -> pd.DataFrame:
    """Flatten one player's history list into a DataFrame with a stable schema."""
    if not history:
        return pd.DataFrame(columns=["player_id", *HISTORY_FIELDS])

    frame = pd.json_normalize(history)
    for column in HISTORY_FIELDS:          # guarantee schema stability
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[HISTORY_FIELDS].copy()
    frame.insert(0, "player_id", player_id)
    return frame


def merge_frames(
    players: pd.DataFrame,
    histories: pd.DataFrame,
    team_lookup: Dict[int, Dict[str, str]],
    season: str,
) -> pd.DataFrame:
    """Join per-gameweek rows onto static player metadata and conform to schema."""
    if histories.empty:
        LOGGER.warning("No gameweek history collected — returning empty frame.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    merged = histories.merge(players, on="player_id", how="left", validate="many_to_one")
    merged = merged.rename(columns={"round": "gameweek", "opponent_team": "opponent_team_id"})

    merged["season"] = season
    merged["gw_cost"] = pd.to_numeric(merged["value"], errors="coerce") / 10.0
    merged["xp"] = pd.NA          # not exposed by element-summary; archive-only

    # Improvement 1: opponent id -> readable short name (season-scoped mapping).
    merged["opponent_team"] = map_opponent_names(merged["opponent_team_id"], team_lookup)

    # `fixture` is the only safe dedupe key: double gameweeks share a `round`.
    merged = merged.drop_duplicates(subset=["player_id", "fixture"], keep="last")

    return finalise(conform_to_schema(merged))


def map_opponent_names(
    ids: pd.Series, team_lookup: Dict[int, Dict[str, str]], key: str = "short_name"
) -> pd.Series:
    """Season-scoped opponent id -> short name ('ARS'), NaN-safe."""
    mapping = {tid: meta.get(key, "UNK") for tid, meta in team_lookup.items()}
    numeric = pd.to_numeric(ids, errors="coerce")
    return numeric.map(mapping).astype("object").where(numeric.notna(), pd.NA)


# --------------------------------------------------------------------------- #
# Live ingestion — orchestration
# --------------------------------------------------------------------------- #

def collect_histories(client: FPLClient, player_ids: List[int], delay: float) -> pd.DataFrame:
    """Loop the element-summary endpoint, sleeping between calls to stay polite."""
    frames: List[pd.DataFrame] = []
    failures: List[int] = []
    total = len(player_ids)

    for index, player_id in enumerate(player_ids, start=1):
        try:
            history = client.fetch_player_history(player_id)
            if history:
                frames.append(normalise_history(history, player_id))
            else:
                failures.append(player_id)
        except Exception as exc:  # never let one player kill a long run
            LOGGER.warning("Unhandled error for player %s: %s", player_id, exc)
            failures.append(player_id)

        if index % 50 == 0 or index == total:
            LOGGER.info("Progress: %d/%d players (%.0f%%)", index, total, 100 * index / total)

        if delay > 0 and index < total:
            time.sleep(delay)  # rate-limit courtesy — avoids IP throttling

    if failures:
        LOGGER.info("%d players returned no history (blanks/new signings/errors).", len(failures))

    if not frames:
        return pd.DataFrame(columns=["player_id", *HISTORY_FIELDS])
    return pd.concat(frames, ignore_index=True)


def run_live_ingestion(client: FPLClient, config: IngestConfig) -> pd.DataFrame:
    """bootstrap -> per-player history -> merge -> conform."""
    start = time.perf_counter()

    bootstrap = client.fetch_bootstrap()
    team_lookup = build_team_lookup(bootstrap["teams"])
    position_lookup = build_position_lookup(bootstrap["element_types"])
    season = derive_season(bootstrap)
    LOGGER.info("Live season detected: %s", season)

    players = build_player_frame(bootstrap["elements"], team_lookup, position_lookup)
    players = select_active_players(players, config.include_unavailable)

    player_ids = players["player_id"].tolist()
    if config.limit:
        player_ids = player_ids[: config.limit]
        LOGGER.info("Limiting run to first %d players.", config.limit)

    LOGGER.info(
        "Fetching gameweek history for %d players (~%.0fs at %.2fs delay) ...",
        len(player_ids), len(player_ids) * config.delay, config.delay,
    )
    histories = collect_histories(client, player_ids, config.delay)
    merged = merge_frames(players.drop(columns=["status"]), histories, team_lookup, season)

    LOGGER.info("Live ingestion done in %.1fs — %d rows.", time.perf_counter() - start, len(merged))
    return merged


# --------------------------------------------------------------------------- #
# Historical ingestion (vaastav/Fantasy-Premier-League archive)
# --------------------------------------------------------------------------- #

class HistoricalIngestor:
    """
    Downloads and harmonises completed-season `merged_gw.csv` files.

    The archive schema drifts between seasons (e.g. `mng_*` columns appeared in
    2024/25, `defensive_contribution` in 2025/26), and team ids are re-issued
    every season, so each season is normalised independently against its own
    `teams.csv` before concatenation.
    """

    def __init__(self, client: FPLClient, include_managers: bool = False) -> None:
        self.client = client
        self.include_managers = include_managers

    # -- fetch ---------------------------------------------------------------
    def fetch_team_lookup(self, season: str) -> Dict[int, Dict[str, str]]:
        """Per-season team id -> names, needed to decode `opponent_team`."""
        url = VAASTAV_TEAMS_URL.format(season=season)
        text = self.client.get_text(url, cache_key=f"vaastav_teams_{season}")
        if not text:
            LOGGER.warning("[%s] teams.csv unavailable — opponent names will be blank.", season)
            return {}
        try:
            frame = pd.read_csv(io.StringIO(text))
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            LOGGER.warning("[%s] Could not parse teams.csv: %s", season, exc)
            return {}
        if not {"id", "name", "short_name"}.issubset(frame.columns):
            LOGGER.warning("[%s] teams.csv missing expected columns.", season)
            return {}
        return {
            int(row.id): {"name": canonical_club(str(row.name_)),
                          "short_name": str(row.short_name)}
            for row in frame.rename(columns={"name": "name_"}).itertuples(index=False)
        }

    def fetch_season_frame(self, season: str) -> Optional[pd.DataFrame]:
        """Download one season's merged_gw.csv. None if unavailable/unparseable."""
        url = VAASTAV_MERGED_GW_URL.format(season=season)
        LOGGER.info("[%s] Downloading merged_gw.csv ...", season)
        text = self.client.get_text(url, cache_key=f"vaastav_merged_gw_{season}")
        if not text:
            LOGGER.warning("[%s] merged_gw.csv unavailable — skipping season.", season)
            return None
        try:
            # encoding_errors guards against the odd mojibake in older files
            frame = pd.read_csv(io.StringIO(text), low_memory=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            LOGGER.warning("[%s] Could not parse merged_gw.csv: %s", season, exc)
            return None
        LOGGER.info("[%s] %d raw rows, %d columns.", season, len(frame), frame.shape[1])
        return frame

    # -- transform -----------------------------------------------------------
    def harmonise(
        self, frame: pd.DataFrame, season: str, team_lookup: Dict[int, Dict[str, str]]
    ) -> pd.DataFrame:
        """Rename, decode ids, and conform a raw archive frame to OUTPUT_COLUMNS."""
        frame = frame.rename(columns=VAASTAV_RENAME).copy()
        frame["season"] = season

        # `round` is the API's gameweek; `GW` is the source file's. They diverge
        # in double gameweeks — prefer `round`, fall back to `GW`.
        if "round" in frame.columns:
            frame["gameweek"] = frame["round"]
            if "GW" in frame.columns:
                frame["gameweek"] = frame["gameweek"].fillna(frame["GW"])
        elif "GW" in frame.columns:
            frame["gameweek"] = frame["GW"]
        else:
            LOGGER.warning("[%s] No round/GW column — cannot assign gameweeks.", season)
            frame["gameweek"] = pd.NA

        # Price at that gameweek (tenths -> £m). `now_cost` stays NaN: today's
        # price applied to a 2023 row would be lookahead.
        frame["gw_cost"] = pd.to_numeric(frame.get("value"), errors="coerce") / 10.0
        frame["now_cost"] = pd.NA

        frame["position"] = frame.get("position", pd.Series("UNK", index=frame.index)).map(
            normalise_position
        )
        if not self.include_managers:
            managers = frame["position"].isin(MANAGER_POSITIONS)
            if managers.any():
                LOGGER.info("[%s] Dropping %d manager rows.", season, int(managers.sum()))
                frame = frame[~managers]

        if "opponent_team_id" in frame.columns and team_lookup:
            frame["opponent_team"] = map_opponent_names(frame["opponent_team_id"], team_lookup)
        else:
            frame["opponent_team"] = pd.NA

        if "team" not in frame.columns:
            # Pre-2023 archives omit `team`; recover it from the opponent map is
            # impossible, so flag loudly rather than silently emitting nulls.
            LOGGER.warning("[%s] No `team` column in archive — team will be blank.", season)
            frame["team"] = pd.NA
            frame["team_id"] = pd.NA
        else:
            # The archive gives the player's club as a name and the opponent as
            # an id. Resolve the name back to that season's id so downstream
            # joins (notably Phase 2's fixture-difficulty step) can match a team
            # to its opponent exactly rather than by string.
            name_to_id = {meta["name"]: tid for tid, meta in team_lookup.items()}
            frame["team_id"] = frame["team"].map(name_to_id)
            unresolved = frame["team_id"].isna().sum()
            if unresolved:
                LOGGER.warning("[%s] %d row(s) have a team name absent from teams.csv.",
                               season, int(unresolved))

        if "player_id" not in frame.columns:
            LOGGER.warning("[%s] No `element` column — player_id will be blank.", season)

        missing = [c for c in OUTPUT_COLUMNS if c not in frame.columns]
        if missing:
            LOGGER.info("[%s] Not present in archive (NaN-filled): %s", season, ", ".join(missing))

        conformed = conform_to_schema(frame)

        # Dedupe on fixture where available (double gameweeks share `round`).
        subset = ["season", "player_id", "gameweek", "kickoff_time"]
        before = len(conformed)
        conformed = conformed.drop_duplicates(subset=subset, keep="last")
        if before != len(conformed):
            LOGGER.info("[%s] Removed %d duplicate rows.", season, before - len(conformed))

        return conformed

    # -- orchestration -------------------------------------------------------
    def run(self, seasons: Sequence[str]) -> pd.DataFrame:
        """Fetch + harmonise every requested season. Missing seasons are skipped."""
        frames: List[pd.DataFrame] = []
        for season in seasons:
            try:
                raw = self.fetch_season_frame(season)
                if raw is None or raw.empty:
                    continue
                team_lookup = self.fetch_team_lookup(season)
                harmonised = self.harmonise(raw, season, team_lookup)
                LOGGER.info(
                    "[%s] Harmonised: %d rows, %d players, GWs %s-%s",
                    season, len(harmonised),
                    harmonised["player_id"].nunique(),
                    _safe_int(harmonised["gameweek"].min()),
                    _safe_int(harmonised["gameweek"].max()),
                )
                frames.append(harmonised)
            except Exception as exc:  # one bad season must not kill the run
                LOGGER.warning("[%s] Unhandled error, skipping season: %s", season, exc)

        if not frames:
            LOGGER.warning("No historical seasons ingested.")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        combined = pd.concat(frames, ignore_index=True)
        return finalise(combined)


def _safe_int(value: Any) -> Any:
    try:
        if pd.isna(value):
            return "-"
        return int(value)
    except (TypeError, ValueError):
        return "-"


# --------------------------------------------------------------------------- #
# Combination + export
# --------------------------------------------------------------------------- #

def combine(historical: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Stack archive + live rows into one frame on the shared schema."""
    parts = [f for f in (historical, live) if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat([conform_to_schema(p.copy()) for p in parts], ignore_index=True)

    # A season may exist in both sources (archive lags the live API mid-season).
    # Keep the live row: it is the fresher of the two.
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["season", "player_id", "gameweek", "kickoff_time"], keep="last"
    )
    if before != len(combined):
        LOGGER.info("Removed %d rows overlapping between archive and live.", before - len(combined))

    return finalise(combined)


def export_csv(frame: pd.DataFrame, output_path: Path, label: str = "output") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(output_path, index=False)
    except OSError as exc:
        raise RuntimeError(f"Failed to write {output_path}: {exc}") from exc
    LOGGER.info("Wrote %s (%s): %d rows x %d cols", output_path, label, len(frame), frame.shape[1])
    return output_path


def summarise(frame: pd.DataFrame) -> None:
    """Log a per-season row/player/gameweek breakdown — cheap sanity check."""
    if frame.empty:
        return
    summary = (
        frame.groupby("season")
        .agg(rows=("player_id", "size"),
             players=("player_id", "nunique"),
             gameweeks=("gameweek", "nunique"),
             total_points=("total_points", "sum"))
        .reset_index()
    )
    LOGGER.info("Per-season summary:\n%s", summary.to_string(index=False))


def run_ingestion(config: IngestConfig) -> Dict[str, pd.DataFrame]:
    """Full pipeline: archive + live -> combined CSV."""
    start = time.perf_counter()
    client = FPLClient(config)

    historical = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not config.skip_historical:
        historical = HistoricalIngestor(client, config.include_managers).run(config.seasons)

    live = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not config.skip_live:
        try:
            live = run_live_ingestion(client, config)
            export_csv(live, config.output_path, "live season")
        except RuntimeError as exc:
            LOGGER.error("Live ingestion failed (%s) — continuing with archive only.", exc)

    combined = combine(historical, live)
    export_csv(combined, config.combined_path, "combined")
    summarise(combined)
    LOGGER.info("Pipeline finished in %.1fs.", time.perf_counter() - start)

    return {"live": live, "historical": historical, "combined": combined}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL gameweek data ingestion (live + archive).")
    parser.add_argument("-o", "--output", default="fpl_raw_gameweek_data.csv",
                        help="Live-season CSV path (default: fpl_raw_gameweek_data.csv)")
    parser.add_argument("-c", "--combined-output", default="fpl_combined_history.csv",
                        help="Combined CSV path (default: fpl_combined_history.csv)")
    parser.add_argument("-s", "--seasons", nargs="+", default=DEFAULT_SEASONS,
                        help=f"Archive seasons to ingest (default: {' '.join(DEFAULT_SEASONS)})")
    parser.add_argument("-d", "--delay", type=float, default=0.5,
                        help="Seconds to sleep between player requests (default: 0.5)")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Only process the first N live players (smoke test)")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds")
    parser.add_argument("--skip-live", action="store_true", help="Archive seasons only")
    parser.add_argument("--skip-historical", action="store_true", help="Live season only")
    parser.add_argument("--include-managers", action="store_true",
                        help="Keep AM/MNG (assistant manager) rows from the archive")
    parser.add_argument("--exclude-unavailable", action="store_true",
                        help="Skip live players with status u/n (not in squad / unavailable)")
    parser.add_argument("--no-cache", action="store_true", help="Disable on-disk caching")
    parser.add_argument("--cache-dir", default=".fpl_cache", help="Directory for cached responses")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.skip_live and args.skip_historical:
        LOGGER.error("Nothing to do: --skip-live and --skip-historical are both set.")
        return 2

    config = IngestConfig(
        output_path=Path(args.output),
        combined_path=Path(args.combined_output),
        seasons=tuple(args.seasons),
        delay=args.delay,
        timeout=args.timeout,
        limit=args.limit,
        include_unavailable=not args.exclude_unavailable,
        include_managers=args.include_managers,
        skip_live=args.skip_live,
        skip_historical=args.skip_historical,
        cache_dir=None if args.no_cache else Path(args.cache_dir),
    )

    try:
        result = run_ingestion(config)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    combined = result["combined"]
    if combined.empty:
        LOGGER.error("Pipeline produced 0 rows — check connectivity and season arguments.")
        return 1

    print("\nCombined preview:")
    preview_cols = ["season", "player_name", "team", "position", "gameweek",
                    "opponent_team", "was_home", "minutes", "total_points",
                    "gw_cost", "expected_goals", "expected_assists"]
    print(combined[preview_cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
