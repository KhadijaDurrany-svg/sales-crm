from __future__ import annotations

import json
from pathlib import Path

from leadsift.bench import BenchConfig, run_benchmark
from leadsift.cli import build_parser, main


def _fast_config() -> BenchConfig:
    return BenchConfig(n_batches=6, n_leads=80, k=10, n_base_records=40, n_boot=300, n_perm=500)


def test_run_benchmark_structure() -> None:
    results = run_benchmark(_fast_config())
    assert results["package"] == "leadsift"
    assert len(results["dataset_checksum_sha256"]) == 64
    assert set(results["ranking"]["methods"]) == {"fit_intent", "raw_score", "recency", "random"}
    assert set(results["dedup"]["methods"]) == {"fuzzy", "exact_match"}


def test_benchmark_fit_intent_beats_baselines() -> None:
    results = run_benchmark(_fast_config())
    methods = results["ranking"]["methods"]
    fit = methods["fit_intent"]["ndcg_at_k_mean"]
    for other in ("raw_score", "recency", "random"):
        assert fit > methods[other]["ndcg_at_k_mean"]


def test_benchmark_dedup_fuzzy_beats_exact() -> None:
    results = run_benchmark(_fast_config())
    d = results["dedup"]["methods"]
    assert d["fuzzy"]["f1_mean"] > d["exact_match"]["f1_mean"]
    assert d["fuzzy"]["f1_mean"] < 1.0  # honest: never perfect


def test_benchmark_is_reproducible() -> None:
    a = run_benchmark(_fast_config())
    b = run_benchmark(_fast_config())
    assert a["dataset_checksum_sha256"] == b["dataset_checksum_sha256"]
    assert a["ranking"]["methods"]["fit_intent"] == b["ranking"]["methods"]["fit_intent"]


def test_cli_bench_no_write(capsys) -> None:  # type: ignore[no-untyped-def]
    # Default config path; --no-write keeps it from touching disk. Just assert it runs.
    rc = main(["bench", "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fit_intent" in out


def test_cli_bench_writes_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "bench_out"
    rc = main(["bench", "--out-dir", str(out_dir)])
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert results["package"] == "leadsift"
    assert manifest["python"]
    assert manifest["dataset_checksum_sha256"] == results["dataset_checksum_sha256"]
    assert "git_sha" in manifest


def test_cli_demo_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["demo", "--seed", "3", "--n", "60", "--top", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Top" in out and "Dedup" in out


def test_parser_requires_subcommand() -> None:
    parser = build_parser()
    try:
        parser.parse_args([])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for missing subcommand")
