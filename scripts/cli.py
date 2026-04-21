"""
Football AI Platform — Unified CLI

A single entry point for all platform operations.

Usage
-----
    python scripts/cli.py run      --input data/raw/fus_FAR.csv
    python scripts/cli.py train
    python scripts/cli.py api
    python scripts/cli.py dashboard
    python scripts/cli.py status
    python scripts/cli.py test
    python scripts/cli.py clean

Each sub-command delegates to the appropriate script or service.

Windows PowerShell:
    $env:PYTHONPATH = "."
    python scripts/cli.py status
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from football_ai.config import settings
from football_ai.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

_ROOT = Path(__file__).parent.parent


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with the project PYTHONPATH set."""
    env = {**__import__("os").environ, "PYTHONPATH": str(_ROOT)}
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(_ROOT),
        env=env,
        check=check,
    )


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """Run the full data + ML + VAEP + tactical pipeline."""
    logger.info("Running end-to-end pipeline ...")
    cmd = [str(_ROOT / "scripts" / "run_pipeline.py")]
    if args.input:
        cmd += ["--input", str(args.input)]
    if args.force:
        cmd.append("--force")
    if args.skip_training:
        cmd.append("--skip-training")
    if args.skip_vaep:
        cmd.append("--skip-vaep")
    if args.skip_tactical:
        cmd.append("--skip-tactical")
    _run(*cmd)


def cmd_train(args: argparse.Namespace) -> None:
    """Train all ML models (requires Phase 4 features to exist)."""
    logger.info("Training ML models ...")
    cmd = [str(_ROOT / "scripts" / "train_models.py")]
    if args.force:
        cmd.append("--force")
    if args.models:
        cmd += ["--models"] + args.models
    _run(*cmd)


def cmd_api(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    logger.info("Starting FastAPI server on port %d ...", args.port)
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "football_ai.api.main:app",
            "--host", args.host,
            "--port", str(args.port),
            "--reload" if args.reload else "",
        ],
        cwd=str(_ROOT),
        check=False,
    )


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Start the Streamlit dashboard."""
    logger.info("Starting Streamlit dashboard on port %d ...", args.port)
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(_ROOT / "football_ai" / "dashboard" / "app.py"),
            "--server.port", str(args.port),
            "--server.address", args.host,
        ],
        cwd=str(_ROOT),
        check=False,
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Show the current pipeline status (which artefacts exist)."""
    from football_ai.ingestion.storage.parquet_store import ParquetStore
    from football_ai.ml.serving.model_registry import ModelRegistry

    store    = ParquetStore()
    registry = ModelRegistry()

    print("\n" + "=" * 55)
    print(f"  {settings.project_name}  v{settings.version}")
    print("=" * 55)

    # Data artefacts
    processed_matches = store.list_processed_matches()
    feature_matches   = store.list_feature_matches()
    vaep_files        = list(store.processed_dir.glob("match_*_vaep.parquet"))
    tactical_files    = list(store.processed_dir.glob("match_*_tactical_report.json"))

    print("\n[files] Data artefacts:")
    print(f"   SPADL processed : {len(processed_matches)} match(es)")
    print(f"   Feature matrices: {len(feature_matches)} match(es)")
    print(f"   VAEP scored     : {len(vaep_files)} match(es)")
    print(f"   Tactical reports: {len(tactical_files)} match(es)")

    # Models
    available = registry.list_available()
    expected  = ["xg", "xt", "p_scores", "p_concedes"]
    print("\n[robot] Model registry:")
    for name in expected:
        status = "OK" if name in available else "FAIL (not trained)"
        print(f"   {name:<14}: {status}")

    # Match details
    if vaep_files:
        print("\n[chart] Match details:")
        for f in sorted(vaep_files):
            stem  = f.stem
            parts = stem.split("_")
            mid   = "_".join(parts[1:-1]) if len(parts) >= 3 else stem
            print(f"   {mid}")

    print("\n" + "=" * 55 + "\n")


def cmd_test(args: argparse.Namespace) -> None:
    """Run the test suite."""
    logger.info("Running %s tests ...", args.suite)
    test_path = {
        "unit":        str(_ROOT / "tests" / "unit"),
        "integration": str(_ROOT / "tests" / "integration"),
        "all":         str(_ROOT / "tests"),
    }.get(args.suite, str(_ROOT / "tests" / "unit"))

    flags = ["-v"] if args.verbose else ["-q"]
    if args.warning_as_error:
        flags.append("-W error::RuntimeWarning")

    _run("-m", "pytest", test_path, *flags, check=False)


def cmd_clean(args: argparse.Namespace) -> None:
    """Remove generated artefacts (Parquet, models, logs)."""
    import shutil

    from football_ai.ingestion.storage.parquet_store import ParquetStore
    store = ParquetStore()

    targets: list[Path] = []

    if args.data or args.all:
        targets += [store.raw_dir, store.processed_dir, store.features_dir]

    if args.models or args.all:
        targets.append(_ROOT / "models")

    if args.logs or args.all:
        targets.append(_ROOT / "logs")

    if not targets:
        print("Nothing to clean. Use --data, --models, --logs, or --all.")
        return

    for t in targets:
        if t.exists():
            shutil.rmtree(t)
            t.mkdir(parents=True, exist_ok=True)
            print(f"Cleaned: {t.relative_to(_ROOT)}")
        else:
            print(f"Already empty: {t.relative_to(_ROOT)}")


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="football-ai",
        description=f"{settings.project_name} -- Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cli.py run --input data/raw/fus_FAR.csv\n"
            "  python scripts/cli.py train --force\n"
            "  python scripts/cli.py api\n"
            "  python scripts/cli.py status\n"
            "  python scripts/cli.py test --suite unit\n"
            "  python scripts/cli.py clean --all\n"
        ),
    )
    subs = p.add_subparsers(dest="command", required=True)

    # run
    run_p = subs.add_parser("run", help="Run end-to-end pipeline")
    run_p.add_argument("--input", type=Path, default=None)
    run_p.add_argument("--force",          action="store_true")
    run_p.add_argument("--skip-training",  action="store_true")
    run_p.add_argument("--skip-vaep",      action="store_true")
    run_p.add_argument("--skip-tactical",  action="store_true")

    # train
    tr_p = subs.add_parser("train", help="Train ML models")
    tr_p.add_argument("--force", action="store_true")
    tr_p.add_argument("--models", nargs="+",
                      choices=["xg", "xt", "p_scores", "p_concedes"])

    # api
    api_p = subs.add_parser("api", help="Start FastAPI server")
    api_p.add_argument("--host",   default="0.0.0.0")
    api_p.add_argument("--port",   type=int, default=8000)
    api_p.add_argument("--reload", action="store_true", default=True)

    # dashboard
    dash_p = subs.add_parser("dashboard", help="Start Streamlit dashboard")
    dash_p.add_argument("--host", default="localhost")
    dash_p.add_argument("--port", type=int, default=8501)

    # status
    subs.add_parser("status", help="Show pipeline status")

    # test
    test_p = subs.add_parser("test", help="Run tests")
    test_p.add_argument("--suite", choices=["unit", "integration", "all"],
                        default="unit")
    test_p.add_argument("--verbose",          "-v", action="store_true")
    test_p.add_argument("--warning-as-error", "-W", action="store_true")

    # clean
    clean_p = subs.add_parser("clean", help="Remove generated artefacts")
    clean_p.add_argument("--data",   action="store_true")
    clean_p.add_argument("--models", action="store_true")
    clean_p.add_argument("--logs",   action="store_true")
    clean_p.add_argument("--all",    action="store_true")

    return p


_DISPATCH = {
    "run":       cmd_run,
    "train":     cmd_train,
    "api":       cmd_api,
    "dashboard": cmd_dashboard,
    "status":    cmd_status,
    "test":      cmd_test,
    "clean":     cmd_clean,
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _DISPATCH[args.command](args)


if __name__ == "__main__":
    main()
