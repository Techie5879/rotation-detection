"""Incremental scale-up runner with stage-by-stage progress and stopping rules."""

from __future__ import annotations

from pathlib import Path

from .experiment import run_experiment
from .logging_utils import tee_output
from .memory_profile import format_memory, snapshot_memory
from .utils import dump_json, load_json, utc_timestamp


def _cap_label(max_pages_per_doc: int | None) -> str:
    if max_pages_per_doc is None:
        return "full"
    return f"cap{max_pages_per_doc}"


def run_curriculum(
    *,
    input_pdfs: list[str],
    input_dir: str | None,
    output_root: str,
    curriculum_name: str | None,
    seed: int,
    dpi: int,
    rotate_probability: float,
    angles: list[int],
    stage_max_pages_per_doc: list[int],
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
    stop_on_regression: bool,
    regression_tolerance: float,
    run_heuristic: bool,
    class_balance: str = "uniform",
    min_val_docs: int = 8,
    min_test_docs: int = 8,
) -> Path:
    """Run staged experiments with increasing page caps and live logs."""
    root = Path(output_root).expanduser().resolve()
    run_name = curriculum_name or f"curriculum_{utc_timestamp()}_seed{seed}"
    curriculum_dir = root / run_name
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    curriculum_log = curriculum_dir / "curriculum.log"

    with tee_output(curriculum_log):
        print(f"[curriculum] run_dir={curriculum_dir}")
        print(f"[curriculum] start {format_memory(snapshot_memory())}")

        stage_caps = stage_max_pages_per_doc or [600, 1000, 2000, 0]
        normalized_caps: list[int | None] = [None if cap <= 0 else int(cap) for cap in stage_caps]

        previous_accuracy: float | None = None
        stage_records: list[dict] = []

        for idx, cap in enumerate(normalized_caps, start=1):
            label = _cap_label(cap)
            stage_name = f"stage{idx:02d}_{label}"
            print(
                f"[curriculum] starting {stage_name} "
                f"max_pages_per_doc={'full' if cap is None else cap}"
            )
            stage_start_mem = snapshot_memory()
            print(f"[curriculum] {stage_name} start {format_memory(stage_start_mem)}")

            stage_exp_dir = run_experiment(
                input_pdfs=input_pdfs,
                input_dir=input_dir,
                output_root=str(curriculum_dir),
                experiment_name=stage_name,
                seed=seed,
                dpi=dpi,
                rotate_probability=rotate_probability,
                angles=angles,
                max_pages_per_doc=cap,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                log_every_pages=log_every_pages,
                class_balance=class_balance,
                min_val_docs=min_val_docs,
                min_test_docs=min_test_docs,
                train_epochs=train_epochs,
                train_batch_size=train_batch_size,
                train_lr=train_lr,
                train_weight_decay=train_weight_decay,
                train_image_size=train_image_size,
                train_val_split=train_val_split,
                train_num_workers=train_num_workers,
                train_device=train_device,
                log_every_batches=log_every_batches,
                run_heuristic=run_heuristic,
            )

            torch_report_path = stage_exp_dir / "report.torch.json"
            dataset_manifest_path = stage_exp_dir / "dataset" / "manifest.json"

            torch_report = load_json(torch_report_path)
            manifest = load_json(dataset_manifest_path)

            torch_acc = float(torch_report.get("accuracy", 0.0))
            heuristic_acc = None
            if run_heuristic:
                heuristic_report_path = stage_exp_dir / "report.heuristic.json"
                heuristic_report = load_json(heuristic_report_path)
                heuristic_acc = float(heuristic_report.get("accuracy", 0.0))
            splits = manifest.get("splits", {})

            stage_record = {
                "stage": stage_name,
                "experiment_dir": str(stage_exp_dir),
                "stage_log": str(stage_exp_dir / "run.log"),
                "max_pages_per_doc": cap,
                "torch_accuracy": torch_acc,
                "heuristic_accuracy": heuristic_acc,
                "split_pages": {name: int(info.get("pages", 0)) for name, info in splits.items()},
                "split_documents": {name: int(info.get("documents", 0)) for name, info in splits.items()},
                "memory_start": stage_start_mem,
                "memory_end": snapshot_memory(),
            }
            stage_records.append(stage_record)

            print(
                f"[curriculum] completed {stage_name} "
                f"torch_acc={torch_acc:.4f}"
                + (f" heuristic_acc={heuristic_acc:.4f}" if heuristic_acc is not None else "")
            )
            print(f"[curriculum] {stage_name} end {format_memory(stage_record['memory_end'])}")

            if stop_on_regression and previous_accuracy is not None:
                if torch_acc + regression_tolerance < previous_accuracy:
                    print(
                        f"[curriculum] stopping early: torch accuracy regressed "
                        f"from {previous_accuracy:.4f} to {torch_acc:.4f}"
                    )
                    break

            previous_accuracy = torch_acc

        summary = {
            "curriculum_dir": str(curriculum_dir),
            "curriculum_log": str(curriculum_log),
            "seed": seed,
            "stages": stage_records,
            "stop_on_regression": stop_on_regression,
            "regression_tolerance": regression_tolerance,
            "run_heuristic": run_heuristic,
            "class_balance": class_balance,
            "min_val_docs": min_val_docs,
            "min_test_docs": min_test_docs,
            "memory_end": snapshot_memory(),
        }
        summary_path = curriculum_dir / "curriculum_summary.json"
        dump_json(summary, summary_path)
        print(f"[curriculum] summary: {summary_path}")
    return summary_path
