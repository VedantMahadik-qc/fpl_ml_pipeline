"""
FPL ML Pipeline — live tracking
===============================

The season replay measures the pipeline on a season that is over. This measures
it on the one being played: an FPL account that follows `fpl_weekly.py` to the
letter, next to a manager picking by hand, against the average of every
manager in the game.

Each run pulls every account's gameweek history from the public FPL API and
writes, for gameweeks from --start-gw that have finished:

    tracking/weekly.csv               points after hits, chip, gameweek rank
    tracking/summary.csv              totals and the running gap to average
    tracking/vs_average.png           cumulative points above the average manager
    tracking/README.md                the table and chart, linked from the README

Team ids stay in tracking/accounts.json, which git ignores: an FPL id opens a
public profile with the manager's name. Everything written out uses labels.

Usage
-----
    python fpl_track.py --add bot=5070003 --add me=4975255     # first time
    python fpl_track.py                                         # after each gameweek
    python fpl_track.py --start-gw 1 -o scratch/                # look further back
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from fpl_optimizer import ensure_printable_console
from fpl_transfers import CHIP_LABELS

LOGGER = logging.getLogger("fpl.track")

BASE_URL = "https://fantasy.premierleague.com/api"
HISTORY_URL = f"{BASE_URL}/entry/{{entry_id}}/history/"

ACCOUNTS_PATH = Path("tracking/accounts.json")
DEFAULT_START_GW = 9          # first deadline planned by the current pipeline


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #

def load_accounts(path: Path = ACCOUNTS_PATH) -> Dict[str, int]:
    if not path.exists():
        return {}
    return {label: int(entry) for label, entry in
            json.loads(path.read_text(encoding="utf-8")).items()}


def save_accounts(accounts: Dict[str, int], path: Path = ACCOUNTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(accounts, indent=2), encoding="utf-8")


def parse_account(item: str) -> tuple:
    label, _, entry = item.partition("=")
    if not label or not entry.isdigit():
        raise SystemExit(f"--add wants LABEL=TEAM_ID, e.g. bot=5070003: {item}")
    return label, int(entry)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def gameweek_table(bootstrap: Dict[str, Any], histories: Dict[str, Dict[str, Any]],
                   start_gw: int) -> pd.DataFrame:
    """
    One row per finished gameweek from start_gw: the average and best score in
    the game, and each account's points after hits, chip and gameweek rank.
    """
    events = {e["id"]: e for e in bootstrap.get("events", [])}
    finished = sorted(gw for gw, e in events.items() if e.get("finished") and gw >= start_gw)
    rows = []
    for gw in finished:
        row: Dict[str, Any] = {"gameweek": gw,
                               "average": events[gw].get("average_entry_score"),
                               "highest": events[gw].get("highest_score")}
        for label, history in histories.items():
            week = next((g for g in history.get("current", []) if g["event"] == gw), None)
            chips = {c["event"]: c["name"] for c in history.get("chips", [])}
            if week is None:
                row[f"{label}_points"] = None
                continue
            row[f"{label}_points"] = week["points"] - week.get("event_transfers_cost", 0)
            row[f"{label}_hits"] = week.get("event_transfers_cost", 0) // 4
            row[f"{label}_chip"] = CHIP_LABELS.get(chips.get(gw), "")
            row[f"{label}_rank"] = week.get("rank")
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        points = table[f"{label}_points"]
        rows.append({
            "account": label,
            "gameweeks": int(points.notna().sum()),
            "points": float(points.sum()),
            "average_manager": float(table.loc[points.notna(), "average"].sum()),
            "vs_average": float((points - table["average"]).sum()),
            "beat_average": int((points > table["average"]).sum()),
            "hits": int(table.get(f"{label}_hits", pd.Series(dtype=float)).sum()),
            "best_week": float(points.max()) if points.notna().any() else None,
            "worst_week": float(points.min()) if points.notna().any() else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_vs_average(table: pd.DataFrame, labels: Sequence[str], path: Path,
                    start_gw: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color=muted, lw=1.2, ls=(0, (4, 3)))
    ax.annotate("average manager", (table["gameweek"].iloc[-1], 0), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=muted)
    for colour, label in zip(COLOURS, labels):
        gap = (table[f"{label}_points"] - table["average"]).cumsum()
        ax.plot(table["gameweek"], gap, color=colour, lw=2, marker="o", ms=5,
                label=f"{label}  {gap.iloc[-1]:+.0f}")
        ax.annotate(f"{label} {gap.iloc[-1]:+.0f}", (table["gameweek"].iloc[-1], gap.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=muted)

    fig.text(0.012, 0.965, "Live season: points above the average manager",
             fontsize=13, color=ink, weight="bold", va="top")
    fig.text(0.012, 0.91, f"Cumulative, after hits, from GW{start_gw}", fontsize=9.5,
             color=muted, va="top")
    ax.set_xlabel("Gameweek", color=muted)
    ax.set_ylabel("Points above average", color=muted)
    ax.tick_params(colors=muted, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_xticks(table["gameweek"])
    ax.legend(frameon=False, loc="upper left", labelcolor=ink, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.9, 0.88))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_markdown(table: pd.DataFrame, summary: pd.DataFrame, labels: Sequence[str],
                   path: Path, start_gw: int) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# Live tracking",
        "",
        f"Points after hits from GW{start_gw}, against the average of every manager in "
        f"the game. Updated {stamp} by `python fpl_track.py`.",
        "",
        "![Points above the average manager](vs_average.png)",
        "",
        "| Account | Gameweeks | Points | Average manager | vs average | Beat average |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.account} | {row.gameweeks} | {row.points:.0f} | "
                     f"{row.average_manager:.0f} | {row.vs_average:+.0f} | "
                     f"{row.beat_average}/{row.gameweeks} |")
    header = "| GW | Average | " + " | ".join(labels) + " |"
    lines += ["", header, "| --- | ---: | " + " | ".join("---:" for _ in labels) + " |"]
    for row in table.itertuples(index=False):
        cells = []
        for label in labels:
            points = getattr(row, f"{label}_points")
            chip = getattr(row, f"{label}_chip", "") or ""
            cells.append("–" if pd.isna(points) else
                         f"{points:.0f}" + (f" ({chip})" if chip else ""))
        lines.append(f"| {row.gameweek} | {row.average:.0f} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(accounts: Dict[str, int], start_gw: int, out: Path) -> Optional[pd.DataFrame]:
    from fpl_ingest import FPLClient, IngestConfig

    client = FPLClient(IngestConfig(cache_dir=None))
    bootstrap = client.fetch_bootstrap()
    histories = {}
    for label, entry in accounts.items():
        history = client.get_json(HISTORY_URL.format(entry_id=entry))
        if not history:
            raise RuntimeError(f"Could not fetch history for '{label}'. Is the team id right?")
        histories[label] = history

    table = gameweek_table(bootstrap, histories, start_gw)
    if table.empty:
        print(f"No finished gameweeks from GW{start_gw} yet; nothing to track.")
        return None

    labels = list(accounts)
    summary = summarise(table, labels)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "weekly.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    plot_vs_average(table, labels, out / "vs_average.png", start_gw)
    write_markdown(table, summary, labels, out / "README.md", start_gw)

    print(summary[["account", "gameweeks", "points", "average_manager", "vs_average",
                   "beat_average", "hits"]].to_string(index=False))
    print(f"\nWrote {out}/weekly.csv, summary.csv, vs_average.png, README.md")
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track FPL accounts against the average manager.")
    parser.add_argument("--add", action="append", default=[], metavar="LABEL=TEAM_ID",
                        help="Add or update an account, e.g. --add bot=5070003")
    parser.add_argument("--remove", action="append", default=[], metavar="LABEL")
    parser.add_argument("--start-gw", type=int, default=DEFAULT_START_GW,
                        help=f"First gameweek to count (default: {DEFAULT_START_GW})")
    parser.add_argument("-o", "--output-dir", default="tracking")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ensure_printable_console()
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    accounts = load_accounts()
    for item in args.add:
        label, entry = parse_account(item)
        accounts[label] = entry
    for label in args.remove:
        accounts.pop(label, None)
    if args.add or args.remove:
        save_accounts(accounts)
        print(f"Tracking: {', '.join(accounts) or 'nobody'} (ids in {ACCOUNTS_PATH})")
    if not accounts:
        LOGGER.error("No accounts yet. Add one with --add bot=TEAM_ID.")
        return 1

    try:
        run(accounts, args.start_gw, Path(args.output_dir))
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
