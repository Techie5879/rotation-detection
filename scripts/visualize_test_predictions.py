#!/usr/bin/env python3
"""Visualize test predictions as input vs corrected-page pairs.

This script is intended for outputs from scripts/test_saved_model.py where
predictions include per-page orientation predictions (and usually truth labels).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rotation_detection.manifest import read_jsonl
from rotation_detection.pdf_ops import rotate_image_clockwise
from rotation_detection.utils import dump_json, utc_timestamp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create visual samples from test predictions (input vs corrected output)."
    )
    parser.add_argument("--dataset-path", required=True, help="Dataset root that contains split labels/images")
    parser.add_argument("--predictions-path", required=True, help="Predictions JSONL from test_saved_model.py")
    parser.add_argument("--split", default="test", help="Split for labels/image lookup (default: test)")
    parser.add_argument(
        "--prediction-field",
        default="predicted_rotation_deg",
        help="Prediction angle field to visualize (default: predicted_rotation_deg)",
    )
    parser.add_argument("--sample-pages", type=int, default=50, help="Number of pages to visualize")
    parser.add_argument("--top-docs", type=int, default=10, help="Number of top-performing docs to include")
    parser.add_argument("--worst-docs", type=int, default=10, help="Number of worst-performing docs to include")
    parser.add_argument("--min-doc-pages", type=int, default=20, help="Minimum pages to rank a doc")
    parser.add_argument("--max-image-height", type=int, default=720, help="Max panel image height")
    parser.add_argument(
        "--include-upright-inputs",
        action="store_true",
        help="Include pages whose true rotation is 0 degrees (default: excluded)",
    )
    parser.add_argument(
        "--include-zero-predictions",
        action="store_true",
        help="Include pages where predicted rotation is 0 degrees (default: excluded)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def _resolve_split_dir(dataset_path: Path, split: str) -> Path:
    split_dir = dataset_path / split
    if (split_dir / "labels.jsonl").exists():
        return split_dir
    if (dataset_path / "labels.jsonl").exists():
        return dataset_path
    raise RuntimeError(
        f"Could not resolve labels.jsonl from dataset_path={dataset_path} split={split}."
    )


def _resolve_image_path(split_dir: Path, dataset_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    split_candidate = split_dir / candidate
    if split_candidate.exists():
        return split_candidate
    root_candidate = dataset_root / candidate
    if root_candidate.exists():
        return root_candidate
    return split_candidate


def _key(doc_id: str, page_index: int) -> tuple[str, int]:
    return str(doc_id), int(page_index)


def _fit_height(image: Image.Image, max_height: int) -> Image.Image:
    if image.height <= max_height:
        return image
    ratio = max_height / float(image.height)
    width = max(1, int(round(image.width * ratio)))
    return image.resize((width, max_height), resample=Image.Resampling.BICUBIC)


def _safe_name(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def _render_assets(
    record: dict[str, Any],
    *,
    assets_dir: Path,
    stem: str,
    max_image_height: int,
) -> dict[str, str]:
    input_path = Path(str(record["image_abs_path"]))
    pred_angle = int(record["predicted_rotation_deg"]) % 360
    corrective_cw = (360 - pred_angle) % 360

    with Image.open(input_path) as source:
        source_rgb = source.convert("RGB")
        corrected = rotate_image_clockwise(source_rgb, corrective_cw)

    left = _fit_height(source_rgb, max_image_height)
    right = _fit_height(corrected, max_image_height)

    title = (
        f"doc={record['doc_id']} page={record['page_index']} bucket={record['bucket']} "
        f"correct={record['correct']}"
    )
    detail = (
        f"true={record['true_rotation_deg']} pred={record['predicted_rotation_deg']} "
        f"confidence={record['confidence']:.4f}"
    )

    margin = 20
    gap = 20
    header_h = 80
    footer_h = 24
    width = margin + left.width + gap + right.width + margin
    height = header_h + max(left.height, right.height) + footer_h

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 8), title, fill=(0, 0, 0))
    draw.text((margin, 30), detail, fill=(0, 0, 0))
    draw.text((margin, 52), "left=input  right=predicted_upright", fill=(70, 70, 70))

    y0 = header_h
    canvas.paste(left, (margin, y0))
    canvas.paste(right, (margin + left.width + gap, y0))

    before_dir = assets_dir / "before"
    after_dir = assets_dir / "after"
    panel_dir = assets_dir / "panel"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = _safe_name(stem)
    before_rel = Path("assets") / "before" / f"{safe_stem}.png"
    after_rel = Path("assets") / "after" / f"{safe_stem}.png"
    panel_rel = Path("assets") / "panel" / f"{safe_stem}.png"

    left.save(assets_dir.parent / before_rel, format="PNG", compress_level=3)
    right.save(assets_dir.parent / after_rel, format="PNG", compress_level=3)
    canvas.save(assets_dir.parent / panel_rel, format="PNG", compress_level=3)

    return {
        "before_rel_path": str(before_rel),
        "after_rel_path": str(after_rel),
        "panel_rel_path": str(panel_rel),
    }


def _write_dashboard_html(summary: dict[str, Any], output_path: Path) -> None:
    summary_json = json.dumps(summary)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Rotation Prediction Viewer</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121a2f;
      --panel2: #18213a;
      --text: #e8edf8;
      --muted: #9fb0d0;
      --ok: #39d98a;
      --bad: #ff6b6b;
      --accent: #6ab3ff;
      --border: #2a3555;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% 0%, #13203f 0%, var(--bg) 45%);
      color: var(--text);
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 20px; }}
    .head {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }}
    .head h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .chip {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 999px; padding: 6px 10px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 280px 1fr; gap: 16px; margin-top: 16px; }}
    .sidebar, .main {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }}
    .section-title {{ margin: 6px 0 10px 0; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }}
    .list {{ margin: 0; padding-left: 16px; font-size: 13px; color: var(--text); }}
    .list li {{ margin: 6px 0; }}
    .controls {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 8px; margin-bottom: 10px; }}
    .controls label {{ display: grid; gap: 4px; font-size: 12px; color: var(--muted); }}
    .controls input, .controls select {{
      border: 1px solid var(--border);
      background: #0f1730;
      color: var(--text);
      border-radius: 8px;
      padding: 7px 8px;
    }}
    .count {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .cards {{ display: grid; gap: 10px; }}
    .card {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 12px; padding: 10px; }}
    .card-head {{ display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-bottom: 8px; }}
    .tag {{ padding: 3px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); }}
    .tag.ok {{ color: var(--ok); border-color: #2f7f57; background: #173625; }}
    .tag.bad {{ color: var(--bad); border-color: #7f3636; background: #381919; }}
    .k {{ color: var(--muted); font-size: 12px; }}
    .v {{ font-size: 13px; }}
    .imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .pane {{ background: #0f1730; border: 1px solid var(--border); border-radius: 10px; padding: 6px; }}
    .pane h4 {{ margin: 2px 0 6px 0; font-size: 12px; color: var(--muted); font-weight: 600; }}
    .pane img {{ width: 100%; height: auto; display: block; border-radius: 8px; }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"head\">
      <h1>Rotation Prediction Viewer</h1>
      <div class=\"meta\">Before/after visualization for sampled test pages.</div>
      <div class=\"stats\" id=\"topStats\"></div>
    </div>

    <div class=\"grid\">
      <aside class=\"sidebar\">
        <div class=\"section-title\">Worst Docs</div>
        <ul class=\"list\" id=\"worstDocs\"></ul>
        <div class=\"section-title\">Best Docs</div>
        <ul class=\"list\" id=\"bestDocs\"></ul>
      </aside>

      <main class=\"main\">
        <div class=\"controls\">
          <label>Bucket
            <select id=\"bucket\">
              <option value=\"all\">all</option>
              <option value=\"worst_doc\">worst_doc</option>
              <option value=\"best_doc\">best_doc</option>
              <option value=\"global_fill\">global_fill</option>
            </select>
          </label>
          <label>Correctness
            <select id=\"correct\">
              <option value=\"all\">all</option>
              <option value=\"correct\">correct</option>
              <option value=\"wrong\">wrong</option>
            </select>
          </label>
          <label>Doc Contains
            <input id=\"doc\" placeholder=\"EW-2-...\" />
          </label>
          <label>Min Confidence
            <input id=\"minConf\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0\" />
          </label>
        </div>
        <div class=\"count\" id=\"count\"></div>
        <div class=\"cards\" id=\"cards\"></div>
      </main>
    </div>
  </div>

  <script>
    const data = {summary_json};
    const samples = data.samples || [];
    const byDoc = new Map((data.doc_stats || []).map((d) => [d.doc_id, d]));

    const topStats = document.getElementById('topStats');
    const worstDocs = document.getElementById('worstDocs');
    const bestDocs = document.getElementById('bestDocs');
    const cards = document.getElementById('cards');
    const count = document.getElementById('count');

    function chip(text) {{
      const span = document.createElement('span');
      span.className = 'chip';
      span.textContent = text;
      return span;
    }}

    topStats.appendChild(chip(`split=${{data.split}}`));
    topStats.appendChild(chip(`samples=${{samples.length}}`));
    topStats.appendChild(chip(`prediction_field=${{data.prediction_field}}`));
    topStats.appendChild(chip(`matched_pages=${{data.matched_pages}}`));

    function renderDocList(container, docIds) {{
      container.innerHTML = '';
      for (const docId of docIds) {{
        const st = byDoc.get(docId);
        const li = document.createElement('li');
        if (st) {{
          li.textContent = `${{docId}} (acc=${{st.accuracy.toFixed(4)}}, pages=${{st.pages}}, err=${{st.errors}})`;
        }} else {{
          li.textContent = docId;
        }}
        container.appendChild(li);
      }}
    }}

    renderDocList(worstDocs, data.worst_docs || []);
    renderDocList(bestDocs, data.top_docs || []);

    const bucketEl = document.getElementById('bucket');
    const correctEl = document.getElementById('correct');
    const docEl = document.getElementById('doc');
    const minConfEl = document.getElementById('minConf');

    function passesFilter(s) {{
      if (bucketEl.value !== 'all' && s.bucket !== bucketEl.value) return false;
      if (correctEl.value === 'correct' && !s.correct) return false;
      if (correctEl.value === 'wrong' && s.correct) return false;
      const q = docEl.value.trim().toLowerCase();
      if (q && !String(s.doc_id).toLowerCase().includes(q)) return false;
      const minConf = Number(minConfEl.value || 0);
      if (Number.isFinite(minConf) && s.confidence < minConf) return false;
      return true;
    }}

    function render() {{
      cards.innerHTML = '';
      const filtered = samples.filter(passesFilter);
      count.textContent = `showing ${{filtered.length}} / ${{samples.length}} samples`;

      for (const s of filtered) {{
        const card = document.createElement('article');
        card.className = 'card';

        const head = document.createElement('div');
        head.className = 'card-head';

        const status = document.createElement('span');
        status.className = `tag ${{s.correct ? 'ok' : 'bad'}}`;
        status.textContent = s.correct ? 'correct' : 'wrong';
        head.appendChild(status);

        const bucket = document.createElement('span');
        bucket.className = 'tag';
        bucket.textContent = s.bucket;
        head.appendChild(bucket);

        const m1 = document.createElement('span');
        m1.innerHTML = `<span class=\"k\">doc</span> <span class=\"v\">${{s.doc_id}}</span>`;
        const m2 = document.createElement('span');
        m2.innerHTML = `<span class=\"k\">page</span> <span class=\"v\">${{s.page_index}}</span>`;
        const m3 = document.createElement('span');
        m3.innerHTML = `<span class=\"k\">true/pred</span> <span class=\"v\">${{s.true_rotation_deg}} / ${{s.predicted_rotation_deg}}</span>`;
        const m4 = document.createElement('span');
        m4.innerHTML = `<span class=\"k\">confidence</span> <span class=\"v\">${{Number(s.confidence).toFixed(4)}}</span>`;
        head.appendChild(m1);
        head.appendChild(m2);
        head.appendChild(m3);
        head.appendChild(m4);

        const imgs = document.createElement('div');
        imgs.className = 'imgs';

        const p1 = document.createElement('div');
        p1.className = 'pane';
        p1.innerHTML = `<h4>Input</h4><img loading=\"lazy\" src=\"${{s.before_rel_path}}\" alt=\"input\" />`;

        const p2 = document.createElement('div');
        p2.className = 'pane';
        p2.innerHTML = `<h4>Predicted Upright</h4><img loading=\"lazy\" src=\"${{s.after_rel_path}}\" alt=\"corrected\" />`;

        imgs.appendChild(p1);
        imgs.appendChild(p2);

        card.appendChild(head);
        card.appendChild(imgs);
        cards.appendChild(card);
      }}
    }}

    [bucketEl, correctEl, docEl, minConfEl].forEach((el) => el.addEventListener('input', render));
    [bucketEl, correctEl].forEach((el) => el.addEventListener('change', render));
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    predictions_path = Path(args.predictions_path).expanduser().resolve()
    split_dir = _resolve_split_dir(dataset_path, args.split)
    dataset_root = split_dir.parent if split_dir != dataset_path else dataset_path
    labels_path = split_dir / "labels.jsonl"

    if not predictions_path.exists():
        raise RuntimeError(f"Predictions file not found: {predictions_path}")
    if not labels_path.exists():
        raise RuntimeError(f"Labels file not found: {labels_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (predictions_path.parent / f"visualize_{utc_timestamp()}").resolve()
    )

    prediction_field = str(args.prediction_field)
    rng = random.Random(int(args.seed))

    labels = read_jsonl(labels_path)
    label_map: dict[tuple[str, int], dict[str, Any]] = {
        _key(str(row["doc_id"]), int(row["page_index"])): row for row in labels
    }

    predictions = read_jsonl(predictions_path)
    if not predictions:
        raise RuntimeError(f"No rows found in predictions file: {predictions_path}")

    joined: list[dict[str, Any]] = []
    missing = 0
    skipped_upright_inputs = 0
    skipped_zero_predictions = 0
    for row in predictions:
        doc_id = str(row["doc_id"])
        page_index = int(row["page_index"])
        k = _key(doc_id, page_index)
        label = label_map.get(k)
        if label is None:
            missing += 1
            continue

        if prediction_field not in row:
            raise RuntimeError(
                f"Prediction field '{prediction_field}' not found in predictions. "
                "Use --prediction-field to select an existing field."
            )

        pred = int(row[prediction_field]) % 360
        true = int(row.get("true_rotation_deg", label.get("rotation_deg", 0))) % 360

        if not bool(args.include_upright_inputs) and true == 0:
            skipped_upright_inputs += 1
            continue
        if not bool(args.include_zero_predictions) and pred == 0:
            skipped_zero_predictions += 1
            continue

        conf = float(row.get("confidence", 0.0))
        image_abs = _resolve_image_path(split_dir, dataset_root, str(label["image_path"]))

        joined.append(
            {
                "doc_id": doc_id,
                "page_index": page_index,
                "true_rotation_deg": true,
                "predicted_rotation_deg": pred,
                "confidence": conf,
                "correct": pred == true,
                "image_rel_path": str(label["image_path"]),
                "image_abs_path": str(image_abs),
            }
        )

    if not joined:
        raise RuntimeError("No prediction rows could be matched to labels.")

    doc_rows: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        doc_rows.setdefault(str(row["doc_id"]), []).append(row)

    doc_stats: list[dict[str, Any]] = []
    for doc_id, rows in doc_rows.items():
        total = len(rows)
        correct = sum(1 for r in rows if bool(r["correct"]))
        accuracy = correct / max(total, 1)
        doc_stats.append(
            {
                "doc_id": doc_id,
                "pages": total,
                "correct": correct,
                "errors": total - correct,
                "accuracy": accuracy,
            }
        )

    eligible = [s for s in doc_stats if int(s["pages"]) >= int(args.min_doc_pages)]
    ranked_pool = eligible if eligible else doc_stats

    worst_docs = [
        str(s["doc_id"])
        for s in sorted(ranked_pool, key=lambda s: (float(s["accuracy"]), -int(s["pages"]), str(s["doc_id"])))
       [: max(0, int(args.worst_docs))]
    ]

    best_docs_raw = [
        str(s["doc_id"])
        for s in sorted(ranked_pool, key=lambda s: (-float(s["accuracy"]), -int(s["pages"]), str(s["doc_id"])))
    ]
    best_docs: list[str] = []
    for doc_id in best_docs_raw:
        if doc_id in worst_docs:
            continue
        best_docs.append(doc_id)
        if len(best_docs) >= max(0, int(args.top_docs)):
            break

    sample_pages = max(1, int(args.sample_pages))
    selected_docs = worst_docs + best_docs
    if not selected_docs:
        selected_docs = [str(s["doc_id"]) for s in sorted(doc_stats, key=lambda s: -int(s["pages"]))[:1]]

    per_doc = max(1, sample_pages // max(1, len(selected_docs)))

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()

    def add_rows(rows: list[dict[str, Any]], bucket: str, limit: int) -> None:
        taken = 0
        for row in rows:
            k = _key(str(row["doc_id"]), int(row["page_index"]))
            if k in selected_keys:
                continue
            item = dict(row)
            item["bucket"] = bucket
            selected.append(item)
            selected_keys.add(k)
            taken += 1
            if taken >= limit:
                break

    for doc_id in worst_docs:
        rows = sorted(
            doc_rows.get(doc_id, []),
            key=lambda r: (
                0 if not bool(r["correct"]) else 1,
                -float(r["confidence"]),
                int(r["page_index"]),
            ),
        )
        add_rows(rows, bucket="worst_doc", limit=per_doc)

    for doc_id in best_docs:
        rows = sorted(
            doc_rows.get(doc_id, []),
            key=lambda r: (
                0 if bool(r["correct"]) else 1,
                -float(r["confidence"]),
                int(r["page_index"]),
            ),
        )
        add_rows(rows, bucket="best_doc", limit=per_doc)

    if len(selected) < sample_pages:
        remaining = [
            row
            for row in sorted(
                joined,
                key=lambda r: (
                    0 if not bool(r["correct"]) else 1,
                    -float(r["confidence"]),
                    int(r["page_index"]),
                ),
            )
            if _key(str(row["doc_id"]), int(row["page_index"])) not in selected_keys
        ]
        add_rows(remaining, bucket="global_fill", limit=sample_pages - len(selected))

    if len(selected) > sample_pages:
        rng.shuffle(selected)
        selected = selected[:sample_pages]

    selected.sort(key=lambda r: (str(r["bucket"]), str(r["doc_id"]), int(r["page_index"])))

    assets_dir = output_dir / "assets"
    for idx, row in enumerate(selected, start=1):
        stem = f"{idx:03d}_{row['bucket']}_{row['doc_id']}_p{int(row['page_index'])+1:05d}"
        rel_paths = _render_assets(
            row,
            assets_dir=assets_dir,
            stem=stem,
            max_image_height=int(args.max_image_height),
        )
        row.update(rel_paths)

    summary = {
        "created_at_utc": utc_timestamp(),
        "dataset_path": str(dataset_path),
        "split": str(args.split),
        "predictions_path": str(predictions_path),
        "prediction_field": prediction_field,
        "missing_label_matches": int(missing),
        "excluded_upright_inputs": int(skipped_upright_inputs),
        "excluded_zero_predictions": int(skipped_zero_predictions),
        "include_upright_inputs": bool(args.include_upright_inputs),
        "include_zero_predictions": bool(args.include_zero_predictions),
        "input_pages": len(predictions),
        "matched_pages": len(joined),
        "selected_pages": len(selected),
        "sample_pages_requested": sample_pages,
        "top_docs": best_docs,
        "worst_docs": worst_docs,
        "doc_stats": sorted(doc_stats, key=lambda s: (float(s["accuracy"]), -int(s["pages"]))),
        "samples": selected,
        "pages_dir": str(output_dir / "assets" / "panel"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary, output_dir / "summary.json")
    _write_dashboard_html(summary, output_dir / "index.html")

    lines = [
        "Prediction Visualization Summary",
        f"predictions_path={predictions_path}",
        f"dataset_path={dataset_path}",
        f"split={args.split}",
        f"prediction_field={prediction_field}",
        f"matched_pages={len(joined)} selected_pages={len(selected)}",
        f"excluded_upright_inputs={skipped_upright_inputs}",
        f"excluded_zero_predictions={skipped_zero_predictions}",
        "",
        "Worst docs:",
    ]
    stats_by_doc = {str(s["doc_id"]): s for s in doc_stats}
    for doc_id in worst_docs:
        s = stats_by_doc.get(doc_id)
        if not s:
            continue
        lines.append(
            f"- {doc_id} acc={float(s['accuracy']):.4f} pages={int(s['pages'])} errors={int(s['errors'])}"
        )
    lines.append("")
    lines.append("Best docs:")
    for doc_id in best_docs:
        s = stats_by_doc.get(doc_id)
        if not s:
            continue
        lines.append(
            f"- {doc_id} acc={float(s['accuracy']):.4f} pages={int(s['pages'])} errors={int(s['errors'])}"
        )
    lines.append("")
    lines.append(f"Dashboard: {output_dir / 'index.html'}")
    lines.append(f"Panels written: {output_dir / 'assets' / 'panel'}")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[visualize] output_dir={output_dir}")
    print(f"[visualize] matched_pages={len(joined)} selected_pages={len(selected)}")
    print(
        f"[visualize] excluded upright_inputs={skipped_upright_inputs} "
        f"zero_predictions={skipped_zero_predictions}"
    )
    print(f"[visualize] top_docs={len(best_docs)} worst_docs={len(worst_docs)}")
    print(f"[visualize] dashboard={output_dir / 'index.html'}")
    print(f"[visualize] panels={output_dir / 'assets' / 'panel'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
