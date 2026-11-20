"""Аугментации, имитирующие съёмку с автомобиля (Агент 1, задача 5).

Идея: используем знания компьютерной графики — физически правдоподобные искажения,
а не просто случайный шум. Все функции принимают и возвращают BGR uint8, не меняют размер кадра
(кроме perspective_warp, которая возвращает и преобразованные bbox).
"""
from __future__ import annotations

import numpy as np

from ..schemas import BBox


def add_motion_blur(img: np.ndarray, length: int = 15, angle_deg: float = 0.0) -> np.ndarray:  # CONTRACT
    """Размытие в движении: свернуть с линейным ядром заданной длины и направления."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §5")


def add_rain(img: np.ndarray, intensity: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Дождь: полупрозрачные наклонные штрихи + лёгкое размытие и снижение контраста."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §5")


def add_glare(img: np.ndarray, strength: float = 0.6, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Блик/засветка: гауссово световое пятно + bloom (размытие ярких областей) на случайной позиции."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §5")


def add_shadow(img: np.ndarray, strength: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Тень от дома/дерева: затемнить случайный многоугольник с мягким краем."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §5")


def add_night(img: np.ndarray, level: float = 0.6, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Сумерки/ночь: гамма + снижение яркости + сдвиг к синему + шум матрицы (пуассон+гаусс)."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §5")


def perspective_warp(  # CONTRACT
    img: np.ndarray, boxes: list[BBox], max_shift: float = 0.08, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, list[BBox]]:
    """Случайная перспективная трансформация (гомография) + пересчёт рамок через ту же матрицу."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §5")


def build_train_augmentations(cfg: dict):  # CONTRACT
    """Вернуть вызываемый объект `aug(img, boxes, class_ids) -> (img, boxes, class_ids)`,
    комбинирующий функции выше со случайными вероятностями (albumentations допускается).
    Параметры вероятностей — в cfg['dataset']['augment'] (добавить секцию в default.yaml).
    Также реализовать CLI-режим: `python -m mgts.data.augment --dataset ... --out ... --factor 2`,
    который офлайн дописывает аугментированные копии только редких классов (для балансировки).
    """
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §5")
