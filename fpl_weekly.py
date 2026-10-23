"""
FPL ML Pipeline — the weekly run, in one command
================================================

Everything a manager needs before a deadline:

    1. refresh   ingest the archive seasons plus every played gameweek of the
                 live season, into a cache folder named for today (a second
                 run the same day reuses it and takes seconds)
    2. features  Phase 2, with next-round odds and the live-season repairs
    3. train     on every completed season, testing on the live season so far
    4. plan      with --team-id: a multi-week transfer plan for YOUR squad,
                 using your real free transfers and selling prices.
                 Without it: the best fresh 15 for the next gameweek.
    5. report    the plan printed here and written to reports/ as a web page

Usage
-----
    python fpl_weekly.py --team-id 1234567          # your plan, 4 weeks ahead
    python fpl_weekly.py                            # best fresh 15
    python fpl_weekly.py --team-id 1234567 --no-refresh --open

Your team id is the number in the URL of your FPL "Points" page:
fantasy.premierleague.com/entry/1234567/event/8
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from fpl_optimizer import POSITIONS, ensure_printable_console
from fpl_transfers import CHIP_LABELS

LOGGER = logging.getLogger("fpl.weekly")

POSITION_ORDER = {p: n for n, p in enumerate(POSITIONS)}


@dataclass
class WeeklyConfig:
    team_id: Optional[int] = None
    horizon: int = 4
    history_seasons: int = 3
    refresh: bool = True
    free_transfers: Optional[int] = None
    chips: bool = True
    bench_weight: float = 0.10
    features_path: Path = Path("fpl_ml_features.csv")
    history_path: Path = Path("fpl_combined_history.csv")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    open_report: bool = False


# --------------------------------------------------------------------------- #
# Seasons and the next deadline
# --------------------------------------------------------------------------- #

def previous_seasons(live: str, count: int) -> List[str]:
    """'2026-27', 3 -> ['2023-24', '2024-25', '2025-26']."""
    start = int(live.split("-")[0])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start - count, start)]


def next_deadline(bootstrap: Dict[str, Any]) -> Dict[str, Any]:
    event = next((e for e in bootstrap.get("events", []) if e.get("is_next")), None)
    if event is None:
        raise RuntimeError("FPL reports no upcoming gameweek — is the season over?")
    return {"gameweek": int(event["id"]), "deadline": event.get("deadline_time")}


def readable_deadline(iso: Optional[str]) -> str:
    """'2026-10-23T17:30:00Z' -> 'Fri 23 Oct, 18:30 local (17:30 UTC)'."""
    if not iso:
        return "unknown"
    moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    local = moment.astimezone()
    return f"{local:%a %d %b, %H:%M} local ({moment:%H:%M} UTC)"


def step(label: str):
    """Print a progress line and return a callable that closes it with the time."""
    print(f"  {label} ...", end="", flush=True)
    started = time.perf_counter()
    return lambda note="": print(f" done in {time.perf_counter() - started:.0f}s"
                                 + (f"  ({note})" if note else ""))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def refresh_data(config: WeeklyConfig, archive: Sequence[str]) -> None:
    from fpl_ingest import IngestConfig, run_ingestion

    # The ingest cache never expires, which is what makes a rerun after a
    # failure cheap. A folder per day keeps that, without replaying last week.
    cache = Path(".fpl_cache") / date.today().isoformat()
    done = step(f"Refreshing data into {cache} (about 6 min on the first run today)")
    run_ingestion(IngestConfig(seasons=tuple(archive), cache_dir=cache,
                               combined_path=config.history_path))
    done()


def build_features(config: WeeklyConfig) -> pd.DataFrame:
    from fpl_features import FeatureConfig, build_features as phase2, export

    done = step("Building features")
    frame = phase2(FeatureConfig(input_path=config.history_path,
                                 output_path=config.features_path))
    export(frame, config.features_path)
    done()
    return frame


def train(config: WeeklyConfig, features: pd.DataFrame, live: str,
          archive: Sequence[str]) -> None:
    from fpl_train import TrainConfig, run_training

    # Test on the live season once it has results; before GW1 is played, hold
    # out the last completed season instead so training still has a test slice.
    if (features["season"] == live).any():
        train_seasons, test_season = list(archive), live
    else:
        train_seasons, test_season = list(archive[:-1]), archive[-1]
    done = step(f"Training on {', '.join(train_seasons)} (test: {test_season})")
    run_training(TrainConfig(input_path=config.features_path, model_dir=config.model_dir,
                             train_seasons=tuple(train_seasons), test_season=test_season,
                             permutation_repeats=0))
    done()


def plan_for_team(config: WeeklyConfig) -> Dict[str, Any]:
    from fpl_transfers import TransferConfig, print_plan, run

    transfer_config = TransferConfig(
        features_path=config.features_path, model_dir=config.model_dir,
        entry_id=config.team_id, horizon=config.horizon,
        free_transfers=config.free_transfers, bench_weight=config.bench_weight,
        use_chips=config.chips,
    )
    done = step(f"Planning {config.horizon} weeks for team {config.team_id}")
    result = run(transfer_config)
    done()
    print_plan(result, transfer_config)
    return result


def fresh_squad(config: WeeklyConfig) -> Dict[str, Any]:
    from fpl_live_inference import LiveConfig, print_live_report, run_live_inference

    live_config = LiveConfig(features_path=config.features_path,
                             model_dir=config.model_dir, bench_weight=config.bench_weight)
    done = step("Picking the best fresh 15")
    outcome = run_live_inference(live_config)
    done()
    print_live_report(outcome["squad"], outcome["gameweek"], live_config, outcome["deadline"])
    return outcome


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --muted: #52514e;
  --line: #e4e3df; --accent: #2a78d6; --out: #c23b3a; --in: #1f8a4c;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #1a1a19; --card: #232322; --ink: #ffffff; --muted: #c3c2b7;
    --line: #3a3a38; --accent: #3987e5; --out: #e66767; --in: #4cc283;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink);
       font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 760px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 0; }}
.sub {{ color: var(--muted); margin: 0 0 20px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px; margin: 0 0 16px; }}
.head {{ display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline;
        margin-bottom: 10px; }}
.meta {{ color: var(--muted); font-size: 13px; }}
.moves {{ margin: 0 0 12px; padding: 0; list-style: none; }}
.moves li {{ padding: 3px 0; }}
.out {{ color: var(--out); }} .in {{ color: var(--in); }}
.note {{ color: var(--muted); font-size: 13px; margin: 0 0 12px; }}
.scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{ text-align: left; color: var(--muted); font-weight: 500; font-size: 12px;
     border-bottom: 1px solid var(--line); padding: 4px 6px; }}
td {{ padding: 5px 6px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.bench td {{ color: var(--muted); }}
tr.divider td {{ color: var(--muted); font-size: 12px; font-weight: 600; padding-top: 12px;
                text-transform: uppercase; letter-spacing: .04em; }}
.tag {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 0 5px;
       border-radius: 4px; background: var(--accent); color: #fff; margin-left: 4px; }}
.total {{ font-size: 15px; }}
.tag.chip {{ font-size: 12px; padding: 1px 7px; margin-left: 8px; vertical-align: 2px; }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{body}
<footer>{footer}</footer>
</main>
</body>
</html>
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _player_rows(squad: pd.DataFrame, prob: bool) -> str:
    """Starters then bench, each in position order."""
    frame = squad.assign(_order=squad["position"].map(POSITION_ORDER)).sort_values(
        ["is_starter", "_order", "predicted_points"], ascending=[False, True, False])
    columns = 7 if prob else 6
    rows, bench_started = [], False
    for row in frame.itertuples():
        if not row.is_starter and not bench_started:
            rows.append(f'<tr class="divider"><td colspan="{columns}">Bench</td></tr>')
            bench_started = True
        tags = ""
        if getattr(row, "is_captain", False):
            tags += '<span class="tag">C</span>'
        if getattr(row, "is_vice_captain", False):
            tags += '<span class="tag">V</span>'
        play = (f'<td class="num">{row.prob_play:.0%}</td>'
                if prob and pd.notna(getattr(row, "prob_play", None)) else
                ('<td class="num">–</td>' if prob else ""))
        rows.append(
            f'<tr class="{"" if row.is_starter else "bench"}">'
            f"<td>{_e(row.position)}</td><td>{_e(row.player_name)}{tags}</td>"
            f"<td>{_e(row.team)}</td><td>{_e(getattr(row, 'fixtures', '') or '')}</td>"
            f'<td class="num">{row.cost:.1f}</td>{play}'
            f'<td class="num">{row.predicted_points:.2f}</td></tr>')
    header = ("<tr><th>Pos</th><th>Player</th><th>Club</th><th>Fixture</th>"
              '<th class="num">£m</th>' + ('<th class="num">Plays</th>' if prob else "")
              + '<th class="num">xPts</th></tr>')
    return f'<div class="scroll"><table>{header}{"".join(rows)}</table></div>'


def render_plan(result: Dict[str, Any]) -> str:
    cards = []
    for index, week in enumerate(result["plan"]):
        moves = "".join(
            f'<li><span class="out">OUT</span> {_e(o)} &rarr; '
            f'<span class="in">IN</span> {_e(i)}</li>'
            for o, i in zip(week["transfers_out"], week["transfers_in"]))
        if week["hits"]:
            cost = f"{week['n_transfers']} transfers, {week['hits']} hit(s) (-{week['hit_cost']:.0f})"
        elif week["n_transfers"]:
            cost = f"{week['n_transfers']} free transfer(s)"
        else:
            cost = "no transfers, roll the free transfer"
        when = "make these now" if index == 0 else "provisional"
        chip = week.get("chip")
        chip_tag = (f'<span class="tag chip">{_e(CHIP_LABELS[chip])}</span>' if chip else "")
        if chip == "freehit":
            cost = "Free Hit: these swaps last one week, your squad returns next week"
        elif chip == "wildcard":
            cost = f"Wildcard: {week['n_transfers']} transfers, all free"
        cards.append(
            f'<section class="card"><div class="head"><h2>GW{week["gameweek"]}{chip_tag}</h2>'
            f'<span class="meta">{when} · {cost} · {week["free_transfers"]} FT · '
            f'bank £{week["bank"]:.1f}m</span>'
            f'<span class="meta total">xPts {week["net_points"]:.1f}</span></div>'
            f'{f"<ul class=moves>{moves}</ul>" if moves else ""}'
            f"{_player_rows(week['squad'], prob=False)}</section>")

    total = sum(w["net_points"] for w in result["plan"])
    hits = sum(w["hits"] for w in result["plan"])
    note = (f'<p class="note">{len(result["plan"])}-week plan, {total:.1f} projected points, '
            f"{hits} hit(s). Only GW{result['plan'][0]['gameweek']} is a commitment; re-run "
            "each week as form, prices and injuries move.</p>")
    return note + "".join(cards)


def render_fresh(outcome: Dict[str, Any]) -> str:
    squad = outcome["squad"].copy()
    starters = squad[squad["is_starter"]]
    captain = squad.loc[squad["is_captain"], "predicted_points"].sum()
    total = starters["predicted_points"].sum() + captain
    note = (f'<p class="note">Best 15 from scratch under £100m: {total:.1f} projected '
            f"points with the captain doubled, squad cost £{squad['cost'].sum():.1f}m. "
            "Useful for a wildcard; for your own team, run with --team-id.</p>")
    return note + f'<section class="card">{_player_rows(squad, prob=True)}</section>'


def write_report(body: str, title: str, subtitle: str, path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = (f"Generated {stamp} by fpl_weekly.py. Projections are a statistical "
              "estimate, not advice; check team news before the deadline.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE.format(title=_e(title), subtitle=_e(subtitle), body=body,
                                footer=_e(footer)), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_weekly(config: WeeklyConfig) -> Path:
    from fpl_ingest import FPLClient, IngestConfig, derive_season

    bootstrap = FPLClient(IngestConfig(cache_dir=None)).fetch_bootstrap()
    live = derive_season(bootstrap)
    archive = previous_seasons(live, config.history_seasons)
    upcoming = next_deadline(bootstrap)
    print(f"\nFPL weekly run: {live}, GW{upcoming['gameweek']} "
          f"(deadline {readable_deadline(upcoming['deadline'])})")

    if config.refresh:
        refresh_data(config, archive)
        features = build_features(config)
        train(config, features, live, archive)
    else:
        print("  Skipping refresh: using the existing feature file and models.")
        if not config.features_path.exists() or not config.model_dir.exists():
            raise FileNotFoundError("No feature file or models yet; run once without "
                                    "--no-refresh.")

    if config.team_id:
        result = plan_for_team(config)
        body = render_plan(result)
        title = f"GW{upcoming['gameweek']} plan for team {config.team_id}"
        name = f"gw{upcoming['gameweek']}_team{config.team_id}.html"
    else:
        outcome = fresh_squad(config)
        body = render_fresh(outcome)
        title = f"GW{upcoming['gameweek']} best fresh 15"
        name = f"gw{upcoming['gameweek']}_fresh15.html"

    subtitle = f"{live} · deadline {readable_deadline(upcoming['deadline'])}"
    path = write_report(body, title, subtitle, config.report_dir / name)
    print(f"\nReport: {path.resolve()}")
    if config.open_report:
        webbrowser.open(path.resolve().as_uri())
    return path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh, retrain and plan for the next FPL deadline in one command.")
    parser.add_argument("--team-id", type=int, default=None,
                        help="Your FPL team id; without it you get the best fresh 15")
    parser.add_argument("-H", "--horizon", type=int, default=4,
                        help="Gameweeks to plan ahead with --team-id (default: 4)")
    parser.add_argument("--free-transfers", type=int, default=None,
                        help="Override the free transfers derived from your history")
    parser.add_argument("--no-chips", action="store_true",
                        help="Plan without Wildcard/Free Hit/Bench Boost/Triple Captain")
    parser.add_argument("--bench-weight", type=float, default=0.10)
    parser.add_argument("--no-refresh", action="store_true",
                        help="Reuse today's data and models; just re-plan")
    parser.add_argument("--history-seasons", type=int, default=3,
                        help="Completed seasons to train on (default: 3)")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--open", action="store_true", help="Open the report in a browser")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ensure_printable_console()
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    config = WeeklyConfig(
        team_id=args.team_id, horizon=args.horizon, history_seasons=args.history_seasons,
        refresh=not args.no_refresh, free_transfers=args.free_transfers,
        chips=not args.no_chips,
        bench_weight=args.bench_weight, report_dir=Path(args.report_dir),
        open_report=args.open,
    )
    try:
        run_weekly(config)
    except (FileNotFoundError, RuntimeError) as exc:
        print()
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print()
        LOGGER.error("Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
