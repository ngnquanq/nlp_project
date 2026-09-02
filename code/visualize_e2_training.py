from __future__ import annotations

import argparse
import json
from pathlib import Path


def _points(rows: list[dict], key: str) -> list[tuple[int, float]]:
    return [(int(row["step"]), float(row[key])) for row in rows if key in row]


def _polyline(
    points: list[tuple[int, float]],
    left: float,
    top: float,
    width: float,
    height: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    def project(point: tuple[int, float]) -> str:
        step, value = point
        x = left + step / x_max * width
        y = top + (y_max - value) / (y_max - y_min) * height
        return f"{x:.1f},{y:.1f}"

    return " ".join(project(point) for point in points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize E2 Trainer loss history as SVG")
    parser.add_argument("trainer_state", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    state = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    train = _points(state["log_history"], "loss")
    validation = _points(state["log_history"], "eval_loss")
    if not train or not validation:
        raise RuntimeError("Trainer state does not contain both training and validation loss")

    x_max = max(step for step, _ in train + validation)
    best_step, best_loss = min(validation, key=lambda item: item[1])
    width, height = 1000, 650
    left, plot_width = 85, 850
    train_top, train_height = 85, 275
    val_top, val_height = 450, 125
    train_min, train_max = 0.7, max(4.2, max(value for _, value in train) + 0.1)
    val_min, val_max = 1.40, 1.61

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#172033}.axis{stroke:#7b8497;stroke-width:1}.grid{stroke:#dce1ea;stroke-width:1}.train{fill:none;stroke:#2563eb;stroke-width:2.5}.val{fill:none;stroke:#dc2626;stroke-width:2.5}.label{font-size:13px}.small{font-size:12px}.title{font-size:22px;font-weight:700}.subtitle{font-size:14px;fill:#596377}</style>',
        '<text x="85" y="36" class="title">E2 Qwen3-8B QLoRA training history</text>',
        f'<text x="85" y="60" class="subtitle">Through step {state["global_step"]}; best validation loss {best_loss:.4f} at step {best_step}</text>',
        '<text x="85" y="78" class="label">Training loss</text>',
    ]

    for value in (1, 2, 3, 4):
        y = train_top + (train_max - value) / (train_max - train_min) * train_height
        svg.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>',
            f'<text x="72" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.1f}</text>',
        ])
    svg.append(f'<polyline points="{_polyline(train, left, train_top, plot_width, train_height, x_max, train_min, train_max)}" class="train"/>')
    svg.extend([
        f'<line x1="{left}" y1="{train_top + train_height}" x2="{left + plot_width}" y2="{train_top + train_height}" class="axis"/>',
        '<text x="85" y="430" class="label">Validation loss (lower is better)</text>',
    ])

    for value in (1.40, 1.45, 1.50, 1.55, 1.60):
        y = val_top + (val_max - value) / (val_max - val_min) * val_height
        svg.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>',
            f'<text x="72" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.2f}</text>',
        ])
    svg.append(f'<polyline points="{_polyline(validation, left, val_top, plot_width, val_height, x_max, val_min, val_max)}" class="val"/>')
    for step, value in validation:
        x = left + step / x_max * plot_width
        y = val_top + (val_max - value) / (val_max - val_min) * val_height
        color = "#16a34a" if step == best_step else "#dc2626"
        radius = 7 if step == best_step else 5
        svg.extend([
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>',
            f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="small">step {step}: {value:.4f}</text>',
        ])

    for step in range(0, int(x_max) + 1, 100):
        x = left + step / x_max * plot_width
        svg.extend([
            f'<line x1="{x:.1f}" y1="{val_top + val_height}" x2="{x:.1f}" y2="{val_top + val_height + 5}" class="axis"/>',
            f'<text x="{x:.1f}" y="{val_top + val_height + 22}" text-anchor="middle" class="small">{step}</text>',
        ])
    svg.extend([
        f'<line x1="{left}" y1="{val_top + val_height}" x2="{left + plot_width}" y2="{val_top + val_height}" class="axis"/>',
        f'<text x="{left + plot_width / 2}" y="625" text-anchor="middle" class="label">Training step</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
