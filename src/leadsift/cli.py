"""Command-line interface for leadsift.

Subcommands:
    bench   run the full benchmark; writes benchmarks/results.json + manifest.json
    demo    score + rank a small seeded batch and run dedup on one batch (prints)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from leadsift import __version__
from leadsift.bench import BenchConfig, run_benchmark
from leadsift.dedup import DedupConfig, detect_duplicates, exact_match_pairs
from leadsift.generate import generate_dedup_records, generate_leads
from leadsift.metrics import prf
from leadsift.ranking import rank_leads
from leadsift.scoring import default_weights, score_lead


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if (out.returncode == 0 and sha) else "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def _manifest(results: dict) -> dict:
    return {
        "package": "leadsift",
        "version": __version__,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _dep_version("numpy"),
        "dataset_checksum_sha256": results.get("dataset_checksum_sha256", "n/a"),
    }


def _dep_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - version lookup is best-effort
        return "n/a"


def _cmd_bench(args: argparse.Namespace) -> int:
    config = BenchConfig()
    results = run_benchmark(config)

    print(json.dumps(results, indent=2))

    if args.no_write:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(_manifest(results), indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'manifest.json'}", file=sys.stderr)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    weights = default_weights()
    ds = generate_leads(seed=args.seed, n=args.n)
    order = rank_leads(ds.leads, ds.horizon_day, weights)

    print(f"Top {min(args.top, len(ds.leads))} leads by fit+intent (seed={args.seed}):\n")
    print(f"{'rank':>4}  {'score':>6}  {'fit':>5}  {'intent':>6}  {'conv':>4}  lead")
    for rank, idx in enumerate(order[: args.top], start=1):
        lead = ds.leads[idx]
        b = score_lead(lead, weights, ds.horizon_day)
        conv = "yes" if lead.converted else "no"
        print(
            f"{rank:>4}  {b.total:>6.3f}  {b.fit:>5.2f}  {b.intent:>6.2f}  {conv:>4}  "
            f"{lead.first_name} {lead.last_name} / {lead.industry} / {lead.title}"
        )

    dd = generate_dedup_records(seed=args.seed, n_base=args.n // 2)
    fuzzy = detect_duplicates(dd.records, DedupConfig())
    exact = exact_match_pairs(dd.records)
    f_m = prf(fuzzy, dd.duplicate_pairs)
    e_m = prf(exact, dd.duplicate_pairs)
    print(f"\nDedup on {len(dd.records)} records ({len(dd.duplicate_pairs)} true duplicate pairs):")
    print(f"  fuzzy        P={f_m.precision:.3f} R={f_m.recall:.3f} F1={f_m.f1:.3f}")
    print(f"  exact-match  P={e_m.precision:.3f} R={e_m.recall:.3f} F1={e_m.f1:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leadsift",
        description="Offline lead prioritization and CRM record deduplication.",
    )
    parser.add_argument("--version", action="version", version=f"leadsift {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("bench", help="run the full benchmark suite")
    p_bench.add_argument(
        "--no-write",
        action="store_true",
        help="print results but do not write benchmarks/ files",
    )
    p_bench.add_argument(
        "--out-dir",
        default="benchmarks",
        help="directory for results.json and manifest.json (default: benchmarks)",
    )
    p_bench.set_defaults(func=_cmd_bench)

    p_demo = sub.add_parser("demo", help="score, rank, and dedup a small seeded batch")
    p_demo.add_argument("--seed", type=int, default=7, help="dataset seed (default: 7)")
    p_demo.add_argument("--n", type=int, default=200, help="leads in the batch (default: 200)")
    p_demo.add_argument("--top", type=int, default=10, help="rows to show (default: 10)")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
