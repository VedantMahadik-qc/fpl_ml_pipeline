"""
FPL ML Pipeline — Phase 1: Data Ingestion
=========================================

Pulls the current-season, player-by-gameweek history from the official
Fantasy Premier League API and flattens it into a single tidy DataFrame
(one row = one player-gameweek), exported to `fpl_raw_gameweek_data.csv`.

Endpoints used
--------------
  bootstrap-static/            -> elements (players), teams, element_types (positions)
  element-summary/{id}/        -> history (current season, per-gameweek rows)

Usage
-----
    python fpl_ingest.py
    python fpl_ingest.py --delay 0.5 --output data/fpl_raw_gameweek_data.csv
    python fpl_ingest.py --limit 25 --log-level DEBUG      # quick smoke test
    python fpl_ingest.py --no-cache                        # ignore on-disk cache

Design notes
------------
* Modular: a thin HTTP client (`FPLClient`) + pure transform functions, so the
  transform layer is unit-testable without touching the network.
* Resilient: session-level retry/backoff on 429/5xx, per-player try/except so a
  single bad response never kills a multi-minute run.
* Polite: configurable inter-request delay (default 0.5s) + a real User-Agent.
* Resumable: raw JSON responses are cached to disk, so a re-run after a failure
  costs seconds instead of ~10 minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

LOGGER = logging.getLogger("fpl.ingest")

# Per-gameweek fields to keep from `history`. Anything missing from a given
# season's payload is created as NaN rather than raising, so the script keeps
# working when the FPL API adds/removes stats between seasons.
HISTORY_FIELDS: List[str] = [
    "round",                          # gameweek number
    "fixture",
    "opponent_team",
    "was_home",
    "kickoff_time",
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

# Columns coerced to numeric after the merge (the API returns several as strings).
NUMERIC_COLS: List[str] = [
    "round", "minutes", "total_points", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_saved",
    "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index", "starts",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution", "value", "selected",
    "transfers_in", "transfers_out", "transfers_balance",
]

# Final column order for the exported CSV.
OUTPUT_COLUMNS: List[str] = [
    "player_id", "player_name", "web_name", "team", "position",
    "now_cost", "gw_cost", "gameweek", "kickoff_time", "opponent_team",
    "was_home", "minutes", "starts", "total_points", "goals_scored",
    "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards",
    "saves", "bonus", "bps", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution", "selected",
    "transfers_in", "transfers_out", "transfers_balance",
]


@dataclass
class IngestConfig:
    """Runtime knobs for a single ingestion run."""

    output_path: Path = Path("fpl_raw_gameweek_data.csv")
    delay: float = 0.5                 # seconds between element-summary calls
    timeout: int = 20                  # per-request timeout (seconds)
    max_retries: int = 4
    limit: Optional[int] = None        # cap players processed (smoke tests)
    include_unavailable: bool = True   # keep injured/suspended players
    cache_dir: Optional[Path] = Path(".fpl_cache")
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

class FPLClient:
    """Thin, retrying wrapper around the public FPL endpoints."""

    def __init__(self, config: IngestConfig) -> None:
        self.config = config
        self.session = self._build_session(config.max_retries)
        if config.cache_dir is not None:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
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
    def _cache_path(self, key: str) -> Optional[Path]:
        if self.config.cache_dir is None:
            return None
        return self.config.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.debug("Cache miss for %s (%s)", key, exc)
            return None

    def _write_cache(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError as exc:
            LOGGER.debug("Could not cache %s (%s)", key, exc)

    # -- requests ------------------------------------------------------------
    def get_json(self, url: str, cache_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """GET a JSON document. Returns None on any unrecoverable failure."""
        if cache_key:
            cached = self._read_cache(cache_key)
            if cached is not None:
                LOGGER.debug("Cache hit: %s", cache_key)
                return cached

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.HTTPError as exc:
            LOGGER.warning("HTTP error for %s: %s", url, exc)
            return None
        except requests.exceptions.Timeout:
            LOGGER.warning("Timed out after %ss: %s", self.config.timeout, url)
            return None
        except requests.exceptions.RequestException as exc:
            LOGGER.warning("Request failed for %s: %s", url, exc)
            return None
        except ValueError as exc:  # includes json.JSONDecodeError
            LOGGER.warning("Non-JSON response from %s: %s", url, exc)
            return None

        if cache_key:
            self._write_cache(cache_key, payload)
        return payload

    def fetch_bootstrap(self) -> Dict[str, Any]:
        """Fetch bootstrap-static. Raises if unavailable — nothing works without it."""
        LOGGER.info("Fetching bootstrap-static ...")
        payload = self.get_json(BOOTSTRAP_URL, cache_key="bootstrap_static")
        if not payload:
            raise RuntimeError("Could not fetch bootstrap-static; aborting ingestion.")
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
# Transform layer (pure functions — no network, easy to unit-test)
# --------------------------------------------------------------------------- #

def build_team_lookup(teams: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """team_id -> {'name', 'short_name'}"""
    return {
        t["id"]: {"name": t.get("name", "Unknown"), "short_name": t.get("short_name", "UNK")}
        for t in teams
    }


def build_position_lookup(element_types: Iterable[Dict[str, Any]]) -> Dict[int, str]:
    """element_type id -> 'GKP' / 'DEF' / 'MID' / 'FWD'"""
    return {
        et["id"]: et.get("singular_name_short") or et.get("singular_name", "UNK")
        for et in element_types
    }


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
                "team_short": team.get("short_name", "UNK"),
                "position": position_lookup.get(el.get("element_type"), "UNK"),
                # now_cost is in tenths of a million (55 -> £5.5m)
                "now_cost": el.get("now_cost", 0) / 10.0,
                "status": el.get("status", "a"),
                "season_minutes": el.get("minutes", 0),
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


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """The API returns xG/ICT stats as strings — make them numeric for modelling."""
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def merge_frames(players: pd.DataFrame, histories: pd.DataFrame) -> pd.DataFrame:
    """Inner-join per-gameweek rows onto static player metadata and tidy up."""
    if histories.empty:
        LOGGER.warning("No gameweek history collected — returning empty frame.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    merged = histories.merge(players, on="player_id", how="left", validate="many_to_one")
    merged = coerce_numeric(merged, NUMERIC_COLS)

    merged = merged.rename(columns={"round": "gameweek"})
    merged["gw_cost"] = merged["value"] / 10.0        # price at that gameweek (£m)

    if "kickoff_time" in merged.columns:
        merged["kickoff_time"] = pd.to_datetime(merged["kickoff_time"], errors="coerce", utc=True)

    merged = merged.drop_duplicates(subset=["player_id", "fixture"], keep="last")

    ordered = [c for c in OUTPUT_COLUMNS if c in merged.columns]
    extras = [c for c in merged.columns if c not in ordered and c not in {"value", "fixture", "status", "season_minutes", "team_short"}]
    merged = merged[ordered + extras]

    return merged.sort_values(["gameweek", "team", "player_name"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
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


def export_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(output_path, index=False)
    except OSError as exc:
        raise RuntimeError(f"Failed to write {output_path}: {exc}") from exc
    LOGGER.info("Wrote %s (%d rows x %d cols)", output_path, len(frame), frame.shape[1])
    return output_path


def run_ingestion(config: IngestConfig) -> pd.DataFrame:
    """Full pipeline: bootstrap -> per-player history -> merge -> CSV."""
    start = time.perf_counter()
    client = FPLClient(config)

    bootstrap = client.fetch_bootstrap()
    team_lookup = build_team_lookup(bootstrap["teams"])
    position_lookup = build_position_lookup(bootstrap["element_types"])

    players = build_player_frame(bootstrap["elements"], team_lookup, position_lookup)
    players = select_active_players(players, config.include_unavailable)

    player_ids = players["player_id"].tolist()
    if config.limit:
        player_ids = player_ids[: config.limit]
        LOGGER.info("Limiting run to first %d players.", config.limit)

    est = len(player_ids) * config.delay
    LOGGER.info("Fetching gameweek history for %d players (~%.0fs at %.2fs delay) ...",
                len(player_ids), est, config.delay)

    histories = collect_histories(client, player_ids, config.delay)
    merged = merge_frames(players, histories)

    LOGGER.info(
        "Done in %.1fs — %d player-gameweek rows, %d unique players, GWs %s-%s",
        time.perf_counter() - start,
        len(merged),
        merged["player_id"].nunique() if not merged.empty else 0,
        int(merged["gameweek"].min()) if not merged.empty else "-",
        int(merged["gameweek"].max()) if not merged.empty else "-",
    )

    export_csv(merged, config.output_path)
    return merged


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL gameweek data ingestion.")
    parser.add_argument("-o", "--output", default="fpl_raw_gameweek_data.csv",
                        help="Output CSV path (default: fpl_raw_gameweek_data.csv)")
    parser.add_argument("-d", "--delay", type=float, default=0.5,
                        help="Seconds to sleep between player requests (default: 0.5)")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Only process the first N players (smoke test)")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds")
    parser.add_argument("--exclude-unavailable", action="store_true",
                        help="Skip players with status u/n (not in squad / unavailable)")
    parser.add_argument("--no-cache", action="store_true", help="Disable on-disk JSON caching")
    parser.add_argument("--cache-dir", default=".fpl_cache", help="Directory for cached JSON")
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

    config = IngestConfig(
        output_path=Path(args.output),
        delay=args.delay,
        timeout=args.timeout,
        limit=args.limit,
        include_unavailable=not args.exclude_unavailable,
        cache_dir=None if args.no_cache else Path(args.cache_dir),
    )

    try:
        frame = run_ingestion(config)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    if frame.empty:
        LOGGER.error("Pipeline produced 0 rows — check connectivity or season start date.")
        return 1

    print("\nPreview:")
    print(frame.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
