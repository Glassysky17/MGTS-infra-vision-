"""Командная строка `mgts` (Агент 4, задача 6). Реализовано как каркас argparse — подключить реальные вызовы."""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
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
    args = build_parser().parse_args(argv)
    # TODO (Агент 4): загрузить конфиг (load_config(args.config)), применить overrides и вызвать:
    #   run        -> pipeline.run_pipeline(...); вывести краткую сводку (сколько объектов, сколько critical/high)
    #   analyze    -> data.analyze.analyze_dataset
    #   prepare    -> data.convert.convert_to_yolo -> data.split.make_split -> data.crops.extract_crops
    #   train-det  -> detection.train.train_detector
    #   train-clf  -> condition.classifier.train_classifier
    #   eval-det   -> detection.evaluate.evaluate_detector
    # Ошибки пользователя (нет папки, нет весов) -> понятное сообщение и код возврата 2, без трейсбека.
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §6")


if __name__ == "__main__":
    sys.exit(main())
