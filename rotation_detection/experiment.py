"""End-to-end runner: dataset -> detect -> evaluate."""

from __future__ import annotations

from pathlib import Path

from .dataset_maker import run_make_dataset
from .detect import run_detection
from .evaluate import run_evaluation
from .logging_utils import tee_output
from .train import run_training
from .utils import dump_json, utc_timestamp


def run_experiment(
    *,
    input_pdfs: list[str],
    input_dir: str | None,
    output_root: str,
    experiment_name: str | None,
    seed: int,
    dpi: int,
    rotate_probability: float,
    angles: list[int],
    max_pages_per_doc: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    log_every_pages: int,
    train_epochs: int,
    train_batch_size: int,
    train_lr: float,
    train_weight_decay: float,
    train_image_size: int,
    train_val_split: float,
    train_num_workers: int,
    train_device: str,
    log_every_batches: int,
    run_heuristic: bool,
) -> Path:
    """Run the full benchmark pipeline with heuristic and torch detectors."""
    root = Path(output_root).expanduser().resolve()
    run_name = experiment_name or f"experiment_{utc_timestamp()}_seed{seed}"
    exp_dir = root / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_log = exp_dir / "run.log"
    with tee_output(run_log):
        print(f"[experiment] run_dir={exp_dir}")

        dataset_dir = run_make_dataset(
            input_pdfs=input_pdfs,
            input_dir=input_dir,
            output_root=str(exp_dir),
            dataset_name="dataset",
            seed=seed,
            dpi=dpi,
            rotate_probability=rotate_probability,
            angles=angles,
            max_pages_per_doc=max_pages_per_doc,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            log_every_pages=log_every_pages,
        )

        root_manifest_path = dataset_dir / "manifest.json"
        test_manifest_path = dataset_dir / "test" / "manifest.json"

        heuristic_preds = exp_dir / "predictions.heuristic.jsonl"
        heuristic_report = exp_dir / "report.heuristic.json"
        if run_heuristic:
            run_detection(
                method="heuristic",
                output_jsonl=str(heuristic_preds),
                dpi=dpi,
                manifest_path=str(test_manifest_path),
                input_pdfs=[],
                input_dir=None,
                checkpoint_path=None,
                device="auto",
                max_pages_per_doc=max_pages_per_doc,
                log_every_pages=log_every_pages,
            )
            run_evaluation(
                manifest_path=str(test_manifest_path),
                predictions_path=str(heuristic_preds),
                output_json=str(heuristic_report),
            )

        checkpoint = exp_dir / "models" / "orientation_cnn.pt"
        run_training(
            manifest_path=str(root_manifest_path),
            labels_path=None,
            checkpoint_path=str(checkpoint),
            epochs=train_epochs,
            batch_size=train_batch_size,
            learning_rate=train_lr,
            weight_decay=train_weight_decay,
            image_size=train_image_size,
            val_split=train_val_split,
            seed=seed,
            num_workers=train_num_workers,
            device=train_device,
            max_samples=None,
            train_split="train",
            val_split_name="val",
            log_every_batches=log_every_batches,
        )

        torch_preds = exp_dir / "predictions.torch.jsonl"
        torch_report = exp_dir / "report.torch.json"
        run_detection(
            method="torch",
            output_jsonl=str(torch_preds),
            dpi=dpi,
            manifest_path=str(test_manifest_path),
            input_pdfs=[],
            input_dir=None,
            checkpoint_path=str(checkpoint),
            device=train_device,
            max_pages_per_doc=max_pages_per_doc,
            log_every_pages=log_every_pages,
        )
        run_evaluation(
            manifest_path=str(test_manifest_path),
            predictions_path=str(torch_preds),
            output_json=str(torch_report),
        )

        summary = {
            "experiment_dir": str(exp_dir),
            "dataset_dir": str(dataset_dir),
            "manifest": str(root_manifest_path),
            "test_manifest": str(test_manifest_path),
            "run_log": str(run_log),
            "run_heuristic": run_heuristic,
            "heuristic_predictions": str(heuristic_preds) if run_heuristic else None,
            "heuristic_report": str(heuristic_report) if run_heuristic else None,
            "torch_checkpoint": str(checkpoint),
            "torch_predictions": str(torch_preds),
            "torch_report": str(torch_report),
        }
        summary_path = exp_dir / "summary.json"
        dump_json(summary, summary_path)
        print(f"Experiment summary: {summary_path}")
    return exp_dir
