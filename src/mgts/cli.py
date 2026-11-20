"""Командная строка `mgts` (Агент 4, задача 6). Реализовано как каркас argparse — подключить реальные вызовы."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from .config import load_config
from .observability import configure_logging, log_context, new_run_id, trace_span

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Собрать CLI-команды MGTS."""
    p = argparse.ArgumentParser(prog="mgts", description="МГТС: мониторинг инфраструктуры по изображениям")
    p.add_argument("--config", default=None, help="доп. YAML-конфиг поверх configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="прогнать пайплайн: картинки -> перечень объектов")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--track", default=None, help="CSV/GPX трек (если координат нет в EXIF)")
    s.add_argument("--backend", default=None, choices=["yolo", "dummy"], help="переопределить detection.backend")

    s = sub.add_parser("analyze", help="отчёт по датасету -> docs/DATASET_REPORT.md")
    s.add_argument("--dataset", required=True)

    s = sub.add_parser("prepare", help="convert -> split -> crops одной командой")
    s.add_argument("--src", required=True)
    s.add_argument("--fmt", default="auto")

    sub.add_parser("train-det", help="обучить детектор")
    sub.add_parser("train-clf", help="обучить классификатор состояния")

    s = sub.add_parser("eval-det", help="метрики детектора на val")
    s.add_argument("--split", default="val")
    return p


def main(argv: list[str] | None = None) -> int:
    """Выполнить CLI-команду и вернуть код завершения."""
    args = build_parser().parse_args(argv)
    configure_logging()
    with log_context(run_id=new_run_id(), command=args.cmd):
        try:
            with trace_span(logger, "cli.command", log_traceback=args.cmd != "run", command=args.cmd):
                return _execute_command(args)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2


def _execute_command(args: argparse.Namespace) -> int:
    with trace_span(logger, "cli.load_config", level=logging.DEBUG, log_traceback=False, config=args.config or "default"):
        cfg = load_config(args.config)
    if args.cmd == "run":
        if args.backend:
            cfg["detection"]["backend"] = args.backend
        from .pipeline import run_pipeline
        records = run_pipeline(args.input, args.output, cfg, track_path=args.track)
        priorities = Counter(record.priority.value for record in records)
        review_count = sum(record.needs_review for record in records)
        print(f"Найдено {len(records)} объектов: {priorities['critical']} critical, "
              f"{priorities['high']} high, {review_count} нужна проверка")
    elif args.cmd == "analyze":
        from .data.analyze import analyze_dataset
        with trace_span(logger, "dataset.analyze", dataset=args.dataset):
            result = analyze_dataset(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.cmd == "prepare":
        from .data.convert import convert_to_yolo
        from .data.crops import extract_crops
        from .data.split import make_split
        dataset_dir = cfg["paths"]["processed_dir"]
        with trace_span(logger, "dataset.convert", source=args.src, output=str(dataset_dir), format=args.fmt):
            convert_to_yolo(args.src, dataset_dir, fmt=args.fmt)
        with trace_span(logger, "dataset.split", output=str(dataset_dir)):
            make_split(dataset_dir, cfg["dataset"]["val_fraction"], cfg["dataset"]["seed"])
        with trace_span(logger, "dataset.extract_crops", output=cfg["paths"]["crops_dir"]):
            extract_crops(dataset_dir, cfg["paths"]["crops_dir"], cfg["dataset"]["crop_padding"])
        print(f"Датасет подготовлен: {dataset_dir}")
    elif args.cmd == "train-det":
        from .detection.train import train_detector
        with trace_span(logger, "training.detector"):
            weights = train_detector(cfg)
        print(f"Веса детектора: {weights}")
    elif args.cmd == "train-clf":
        from .condition.classifier import train_classifier
        with trace_span(logger, "training.classifier"):
            weights = train_classifier(cfg)
        print(f"Веса классификатора: {weights}")
    elif args.cmd == "eval-det":
        from .detection.evaluate import evaluate_detector
        with trace_span(logger, "evaluation.detector", split=args.split):
            result = evaluate_detector(cfg, args.split)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
