"""Command-line entry point for Layer 4.

    python -m src.prediction.cli train
    python -m src.prediction.cli score AAPL
    python -m src.prediction.cli score --all
    python -m src.prediction.cli report
"""

from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.db.database import init_db, session_scope
from src.logging_config import setup_logging
from src.prediction import repository
from src.prediction.predict import ModelUnavailable, load_model, score


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=settings.log_level)

    parser = argparse.ArgumentParser(
        prog="prediction",
        parents=[common],
        description="Train and apply the financial-distress probability model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", parents=[common], help="Fit on the American bankruptcy dataset")
    train.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="isotonic")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--data-dir", default=None, help="Where the training CSV is cached")
    train.add_argument("--no-download", action="store_true", help="Fail if the dataset is absent")

    score_cmd = sub.add_parser("score", parents=[common], help="Score company-years and store them")
    score_cmd.add_argument("tickers", nargs="*")
    score_cmd.add_argument("--all", action="store_true", dest="score_all")
    score_cmd.add_argument("--year", type=int, help="Only this fiscal year")
    score_cmd.add_argument("--no-store", action="store_true", help="Print without writing")

    sub.add_parser("report", parents=[common], help="Show the trained model's metrics")
    return parser


def _cmd_train(args) -> int:
    from src.prediction.dataset import load_training_frame
    from src.prediction.model import train_model

    frame = load_training_frame(args.data_dir, download=not args.no_download)
    model = train_model(
        frame, calibration=args.calibration, seed=args.seed, folds=args.folds
    )
    model.save(settings.distress_model_path)
    _print_report(model)
    return 0


def _print_report(model) -> None:
    m = model.metrics
    print(f"\n{'=' * 66}\n{model.model_name} v{model.model_version}\n{'=' * 66}")
    print(f"trained_at        {model.trained_at}")
    print(f"training rows     {model.n_train:,} ({model.n_positive:,} insolvent, "
          f"{model.prevalence:.2%})")
    print(f"\ncross-validated ({m['cv_folds']} folds, out of fold):")
    print(f"  model     ROC-AUC {m['model']['roc_auc']:.3f}   "
          f"PR-AUC {m['model']['pr_auc']:.3f}   Brier {m['model']['brier']:.4f}")
    print(f"  baseline  ROC-AUC {m['baseline']['roc_auc']:.3f}   "
          f"PR-AUC {m['baseline']['pr_auc']:.3f}   (naive linear rule, same features)")
    print(f"  PR-AUC lift over baseline: {m['lift_pr_auc']:+.3f}")
    print("\ntop features (permutation importance):")
    for driver in model.drivers[:6]:
        print(f"  {driver['feature']:<28} {driver['importance']:.4f}")
    print(f"\nrisk bands: " + ", ".join(f"<{c:.2f} {label}" for c, label in model.bands))


def _cmd_score(args) -> int:
    try:
        model = load_model()
    except ModelUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1

    init_db()
    with session_scope() as session:
        tickers = (
            repository.all_scored_tickers(session) if args.score_all else [t.upper() for t in args.tickers]
        )
        if not tickers:
            print("Pass tickers or --all.", file=sys.stderr)
            return 2

        grand_total = 0
        for ticker in tickers:
            rows = repository.build_feature_rows(session, ticker)
            if args.year is not None:
                rows = [r for r in rows if r.fiscal_year == args.year]
            if not rows:
                print(f"  {ticker:<8} no statements on file", file=sys.stderr)
                continue

            estimates = [(row, score(row.features)) for row in rows]
            if not args.no_store:
                grand_total += repository.save_estimates(session, ticker, estimates)

            latest_row, latest = max(estimates, key=lambda pair: pair[0].fiscal_year)
            if latest.is_scored:
                factors = ", ".join(f["feature"] for f in latest.factors) or "-"
                print(f"  {ticker:<8} FY{latest_row.fiscal_year}  "
                      f"p={latest.probability:.3f}  {latest.risk_band:<9} [{factors}]")
            else:
                print(f"  {ticker:<8} FY{latest_row.fiscal_year}  not scored - {latest.reason}")

        if not args.no_store:
            print(f"\nStored {grand_total} estimates for {len(tickers)} companies "
                  f"(model {model.model_name} v{model.model_version}).")
    return 0


def _cmd_report(args) -> int:
    try:
        model = load_model()
    except ModelUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    _print_report(model)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return {
        "train": _cmd_train,
        "score": _cmd_score,
        "report": _cmd_report,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
