"""CLI entrypoint for the PDF rotation detection pipeline."""

from __future__ import annotations

import argparse

from rotation_detection.dataset_maker import run_make_dataset
from rotation_detection.detect import run_detection
from rotation_detection.evaluate import run_evaluation
from rotation_detection.experiment import run_experiment
from rotation_detection.curriculum import run_curriculum
from rotation_detection.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rotation-detection",
        description="Metadata-agnostic PDF orientation pipeline (no Tesseract OSD).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ds = subparsers.add_parser("make-dataset", help="Create synthetic rotated dataset from PDFs")
    ds.add_argument("--input-dir", default="test_pdfs", help="Directory scanned recursively for PDFs")
    ds.add_argument("--input-pdf", action="append", default=[], help="Explicit PDF path (repeatable)")
    ds.add_argument("--output-root", default="datasets", help="Root directory for generated datasets")
    ds.add_argument("--dataset-name", default=None, help="Dataset folder name (default auto-generated)")
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--dpi", type=int, default=144)
    ds.add_argument("--rotate-probability", type=float, default=1.0)
    ds.add_argument("--angles", nargs="+", type=int, default=[0, 90, 180, 270])
    ds.add_argument("--max-pages-per-doc", type=int, default=None)
    ds.add_argument("--train-ratio", type=float, default=0.8)
    ds.add_argument("--val-ratio", type=float, default=0.1)
    ds.add_argument("--test-ratio", type=float, default=0.1)
    ds.add_argument("--class-balance", choices=["random", "uniform"], default="uniform")
    ds.add_argument("--min-val-docs", type=int, default=8)
    ds.add_argument("--min-test-docs", type=int, default=8)
    ds.add_argument("--log-every-pages", type=int, default=200)

    detect = subparsers.add_parser("detect", help="Predict orientation for each page")
    detect.add_argument("--method", choices=["heuristic", "torch"], default="heuristic")
    detect.add_argument("--manifest-path", default=None, help="Use dataset manifest as detection target list")
    detect.add_argument("--input-dir", default=None, help="Directory scanned recursively for PDFs")
    detect.add_argument("--input-pdf", action="append", default=[], help="Explicit PDF path (repeatable)")
    detect.add_argument("--checkpoint-path", default=None, help="Torch checkpoint path")
    detect.add_argument("--device", default="auto", help="Torch device: auto, cpu, mps, cuda")
    detect.add_argument("--dpi", type=int, default=144)
    detect.add_argument("--max-pages-per-doc", type=int, default=None)
    detect.add_argument("--log-every-pages", type=int, default=200)
    detect.add_argument("--output-jsonl", required=True)

    train = subparsers.add_parser("train", help="Train torch orientation classifier")
    train.add_argument("--manifest-path", default=None)
    train.add_argument("--labels-path", default=None)
    train.add_argument("--checkpoint-path", required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--image-size", type=int, default=320)
    train.add_argument("--val-split", type=float, default=0.2)
    train.add_argument("--train-split", default="train", help="Split name for training labels in root manifest")
    train.add_argument("--val-split-name", default="val", help="Split name for validation labels in root manifest")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--max-samples", type=int, default=None)
    train.add_argument("--log-every-batches", type=int, default=50)
    train.add_argument("--early-stopping-patience", type=int, default=2)
    train.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against manifest labels")
    evaluate.add_argument("--manifest-path", required=True)
    evaluate.add_argument("--predictions-path", required=True)
    evaluate.add_argument("--output-json", required=True)

    exp = subparsers.add_parser("run-experiment", help="Run dataset + heuristic + torch benchmark")
    exp.add_argument("--input-dir", default="test_pdfs")
    exp.add_argument("--input-pdf", action="append", default=[])
    exp.add_argument("--output-root", default="runs")
    exp.add_argument("--experiment-name", default=None)
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument("--dpi", type=int, default=144)
    exp.add_argument("--rotate-probability", type=float, default=1.0)
    exp.add_argument("--angles", nargs="+", type=int, default=[0, 90, 180, 270])
    exp.add_argument("--max-pages-per-doc", type=int, default=None)
    exp.add_argument("--train-ratio", type=float, default=0.8)
    exp.add_argument("--val-ratio", type=float, default=0.1)
    exp.add_argument("--test-ratio", type=float, default=0.1)
    exp.add_argument("--class-balance", choices=["random", "uniform"], default="uniform")
    exp.add_argument("--min-val-docs", type=int, default=8)
    exp.add_argument("--min-test-docs", type=int, default=8)
    exp.add_argument("--log-every-pages", type=int, default=200)
    exp.add_argument("--train-epochs", type=int, default=10)
    exp.add_argument("--train-batch-size", type=int, default=64)
    exp.add_argument("--train-lr", type=float, default=3e-4)
    exp.add_argument("--train-weight-decay", type=float, default=1e-4)
    exp.add_argument("--train-image-size", type=int, default=320)
    exp.add_argument("--train-val-split", type=float, default=0.2)
    exp.add_argument("--train-num-workers", type=int, default=0)
    exp.add_argument("--train-device", default="auto")
    exp.add_argument("--log-every-batches", type=int, default=50)
    exp.add_argument("--skip-heuristic", action="store_true")

    cur = subparsers.add_parser("run-curriculum", help="Run staged scale-up experiments with progress logs")
    cur.add_argument("--input-dir", default="test_pdfs")
    cur.add_argument("--input-pdf", action="append", default=[])
    cur.add_argument("--output-root", default="runs")
    cur.add_argument("--curriculum-name", default=None)
    cur.add_argument("--seed", type=int, default=42)
    cur.add_argument("--dpi", type=int, default=96)
    cur.add_argument("--rotate-probability", type=float, default=1.0)
    cur.add_argument("--angles", nargs="+", type=int, default=[0, 90, 180, 270])
    cur.add_argument("--stage-max-pages-per-doc", nargs="+", type=int, default=[600, 1000, 2000, 0])
    cur.add_argument("--train-ratio", type=float, default=0.8)
    cur.add_argument("--val-ratio", type=float, default=0.1)
    cur.add_argument("--test-ratio", type=float, default=0.1)
    cur.add_argument("--class-balance", choices=["random", "uniform"], default="uniform")
    cur.add_argument("--min-val-docs", type=int, default=8)
    cur.add_argument("--min-test-docs", type=int, default=8)
    cur.add_argument("--log-every-pages", type=int, default=200)
    cur.add_argument("--train-epochs", type=int, default=5)
    cur.add_argument("--train-batch-size", type=int, default=96)
    cur.add_argument("--train-lr", type=float, default=3e-4)
    cur.add_argument("--train-weight-decay", type=float, default=1e-4)
    cur.add_argument("--train-image-size", type=int, default=256)
    cur.add_argument("--train-val-split", type=float, default=0.2)
    cur.add_argument("--train-num-workers", type=int, default=0)
    cur.add_argument("--train-device", default="auto")
    cur.add_argument("--log-every-batches", type=int, default=20)
    cur.add_argument("--stop-on-regression", action="store_true")
    cur.add_argument("--regression-tolerance", type=float, default=0.02)
    cur.add_argument("--skip-heuristic", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "make-dataset":
        run_make_dataset(
            input_pdfs=args.input_pdf,
            input_dir=args.input_dir,
            output_root=args.output_root,
            dataset_name=args.dataset_name,
            seed=args.seed,
            dpi=args.dpi,
            rotate_probability=args.rotate_probability,
            angles=args.angles,
            max_pages_per_doc=args.max_pages_per_doc,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            log_every_pages=args.log_every_pages,
            class_balance=args.class_balance,
            min_val_docs=args.min_val_docs,
            min_test_docs=args.min_test_docs,
        )
        return 0

    if args.command == "detect":
        run_detection(
            method=args.method,
            output_jsonl=args.output_jsonl,
            dpi=args.dpi,
            manifest_path=args.manifest_path,
            input_pdfs=args.input_pdf,
            input_dir=args.input_dir,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            max_pages_per_doc=args.max_pages_per_doc,
            log_every_pages=args.log_every_pages,
        )
        return 0

    if args.command == "train":
        run_training(
            manifest_path=args.manifest_path,
            labels_path=args.labels_path,
            checkpoint_path=args.checkpoint_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            val_split=args.val_split,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
            max_samples=args.max_samples,
            train_split=args.train_split,
            val_split_name=args.val_split_name,
            log_every_batches=args.log_every_batches,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
        return 0

    if args.command == "evaluate":
        run_evaluation(
            manifest_path=args.manifest_path,
            predictions_path=args.predictions_path,
            output_json=args.output_json,
        )
        return 0

    if args.command == "run-experiment":
        run_experiment(
            input_pdfs=args.input_pdf,
            input_dir=args.input_dir,
            output_root=args.output_root,
            experiment_name=args.experiment_name,
            seed=args.seed,
            dpi=args.dpi,
            rotate_probability=args.rotate_probability,
            angles=args.angles,
            max_pages_per_doc=args.max_pages_per_doc,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            log_every_pages=args.log_every_pages,
            class_balance=args.class_balance,
            min_val_docs=args.min_val_docs,
            min_test_docs=args.min_test_docs,
            train_epochs=args.train_epochs,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            train_weight_decay=args.train_weight_decay,
            train_image_size=args.train_image_size,
            train_val_split=args.train_val_split,
            train_num_workers=args.train_num_workers,
            train_device=args.train_device,
            log_every_batches=args.log_every_batches,
            run_heuristic=(not args.skip_heuristic),
        )
        return 0

    if args.command == "run-curriculum":
        run_curriculum(
            input_pdfs=args.input_pdf,
            input_dir=args.input_dir,
            output_root=args.output_root,
            curriculum_name=args.curriculum_name,
            seed=args.seed,
            dpi=args.dpi,
            rotate_probability=args.rotate_probability,
            angles=args.angles,
            stage_max_pages_per_doc=args.stage_max_pages_per_doc,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            log_every_pages=args.log_every_pages,
            class_balance=args.class_balance,
            min_val_docs=args.min_val_docs,
            min_test_docs=args.min_test_docs,
            train_epochs=args.train_epochs,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            train_weight_decay=args.train_weight_decay,
            train_image_size=args.train_image_size,
            train_val_split=args.train_val_split,
            train_num_workers=args.train_num_workers,
            train_device=args.train_device,
            log_every_batches=args.log_every_batches,
            stop_on_regression=args.stop_on_regression,
            regression_tolerance=args.regression_tolerance,
            run_heuristic=(not args.skip_heuristic),
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
