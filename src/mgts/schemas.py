"""ЗАМОРОЖЕННЫЕ контракты данных (CONTRACT). Общие для всех модулей.

Не зависит от numpy/cv2/torch — только стандартная библиотека.
Изменения — только через tasks/CONTRACT_REQUESTS.md.

Поток данных:
    FrameMeta ──► Detection ──► ConditionAssessment ──► Observation ──► ObjectRecord (после дедупликации)
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Optional

SCHEMA_VERSION = "1.0"


class Severity(str, Enum):
    OK = "ok"                  # дефектов не найдено
    WARNING = "warning"        # есть признаки, желательна проверка
    DEFECT = "defect"          # состояние неудовлетворительное, нужен ремонт/проверка
    UNCERTAIN = "uncertain"    # модель не уверена -> ручная проверка


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


PRIORITY_ORDER = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]

LOCATION_SOURCES = ("exif", "track", "csv", "synthetic", "none")


# ───────────────────────── Геометрия ─────────────────────────
@dataclass(frozen=True)
class BBox:
    """Рамка в АБСОЛЮТНЫХ пикселях исходного кадра: (x1, y1) левый верх, (x2, y2) правый низ."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Точка основания объекта (для проекции на землю)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def clip(self, img_w: int, img_h: int) -> "BBox":
        return BBox(
            min(max(self.x1, 0.0), img_w), min(max(self.y1, 0.0), img_h),
            min(max(self.x2, 0.0), img_w), min(max(self.y2, 0.0), img_h),
        )

    def pad(self, frac: float, img_w: int, img_h: int) -> "BBox":
        """Расширить рамку на долю frac от её размеров и обрезать по кадру."""
        dw, dh = self.width * frac, self.height * frac
        return BBox(self.x1 - dw, self.y1 - dh, self.x2 + dw, self.y2 + dh).clip(img_w, img_h)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_xywhn(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """YOLO-формат: (xc, yc, w, h), нормировано на [0, 1]."""
        cx, cy = self.center
        return (cx / img_w, cy / img_h, self.width / img_w, self.height / img_h)

    @classmethod
    def from_xywhn(cls, xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> "BBox":
        bw, bh = w * img_w, h * img_h
        cx, cy = xc * img_w, yc * img_h
        return cls(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, v: list[float]) -> "BBox":
        return cls(*[float(x) for x in v])


# ───────────────────────── Кадр ─────────────────────────
@dataclass
class FrameMeta:
    """Метаданные кадра. lat/lon — позиция АВТОМОБИЛЯ в момент съёмки (не объекта)."""
    frame_id: str                          # уникален в рамках запуска, обычно имя файла без расширения
    path: str
    width: int = 0
    height: int = 0
    lat: Optional[float] = None
    lon: Optional[float] = None
    heading_deg: Optional[float] = None    # курс: 0=север, 90=восток
    timestamp: Optional[str] = None        # ISO 8601
    location_source: str = "none"          # одно из LOCATION_SOURCES


# ───────────────────────── Детекция ─────────────────────────
@dataclass
class Detection:
    frame_id: str
    class_id: int
    class_name: str
    score: float                           # уверенность детектора 0..1
    bbox: BBox


# ───────────────────────── Состояние ─────────────────────────
@dataclass
class DefectFinding:
    code: str                              # ключ из configs/classes.yaml -> defects
    score: float                           # 0..1, насколько признак выражен/вероятен
    source: str = "rule"                   # geometry | classifier | rule
    evidence: str = ""                     # человекочитаемое основание («наклон 9.3° > 8°»)


@dataclass
class ConditionAssessment:
    severity: Severity = Severity.OK
    score: float = 0.0                     # 0..1, «степень неисправности» (1 = совсем плохо)
    findings: list[DefectFinding] = field(default_factory=list)
    needs_review: bool = False             # True -> показать оператору

    @property
    def top_finding(self) -> Optional[DefectFinding]:
        return max(self.findings, key=lambda f: f.score) if self.findings else None


@dataclass
class Observation:
    """Одно наблюдение объекта на одном кадре (до дедупликации)."""
    frame_id: str
    detection: Detection
    assessment: ConditionAssessment
    sharpness: float = 0.0                 # blur_score кадра/вырезки
    est_lat: Optional[float] = None        # оценка позиции ОБЪЕКТА
    est_lon: Optional[float] = None
    location_source: str = "none"
    crop_path: Optional[str] = None
    frame_path: Optional[str] = None


# ───────────────────────── Итоговая запись ─────────────────────────
CSV_COLUMNS = [
    "object_id", "class_name", "priority", "severity", "condition_score", "needs_review",
    "defects", "description", "lat", "lon", "location_source",
    "confidence", "n_observations", "best_frame_id", "crop_path",
]


@dataclass
class ObjectRecord:
    """Уникальный объект инфраструктуры в итоговом перечне."""
    object_id: str                         # «OBJ-0001»
    class_name: str
    severity: Severity
    condition_score: float
    needs_review: bool
    priority: Priority
    findings: list[DefectFinding] = field(default_factory=list)
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_source: str = "none"
    confidence: float = 0.0                # уверенность детектора лучшего наблюдения
    n_observations: int = 1
    best_frame_id: str = ""
    best_bbox: Optional[BBox] = None
    frame_ids: list[str] = field(default_factory=list)
    crop_path: Optional[str] = None
    frame_path: Optional[str] = None

    @property
    def description(self) -> str:
        return "; ".join(f"{f.code}: {f.evidence}" if f.evidence else f.code for f in self.findings)

    def to_row(self) -> dict[str, Any]:
        """Плоская строка для CSV (колонки = CSV_COLUMNS)."""
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "priority": self.priority.value,
            "severity": self.severity.value,
            "condition_score": round(self.condition_score, 3),
            "needs_review": self.needs_review,
            "defects": ";".join(f.code for f in self.findings),
            "description": self.description,
            "lat": self.lat,
            "lon": self.lon,
            "location_source": self.location_source,
            "confidence": round(self.confidence, 3),
            "n_observations": self.n_observations,
            "best_frame_id": self.best_frame_id,
            "crop_path": self.crop_path,
        }


# ───────────────────────── (де)сериализация ─────────────────────────
def to_jsonable(obj: Any) -> Any:
    """Рекурсивно превращает dataclass/Enum/BBox в JSON-совместимые типы."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, BBox):
        return obj.to_list()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def detection_from_dict(d: dict) -> Detection:
    return Detection(d["frame_id"], int(d["class_id"]), d["class_name"], float(d["score"]), BBox.from_list(d["bbox"]))


def finding_from_dict(d: dict) -> DefectFinding:
    return DefectFinding(d["code"], float(d["score"]), d.get("source", "rule"), d.get("evidence", ""))


def assessment_from_dict(d: dict) -> ConditionAssessment:
    return ConditionAssessment(
        Severity(d.get("severity", "ok")), float(d.get("score", 0.0)),
        [finding_from_dict(f) for f in d.get("findings", [])], bool(d.get("needs_review", False)),
    )


def observation_from_dict(d: dict) -> Observation:
    return Observation(
        frame_id=d["frame_id"], detection=detection_from_dict(d["detection"]),
        assessment=assessment_from_dict(d["assessment"]), sharpness=float(d.get("sharpness", 0.0)),
        est_lat=d.get("est_lat"), est_lon=d.get("est_lon"), location_source=d.get("location_source", "none"),
        crop_path=d.get("crop_path"), frame_path=d.get("frame_path"),
    )


def object_record_from_dict(d: dict) -> ObjectRecord:
    return ObjectRecord(
        object_id=d["object_id"], class_name=d["class_name"], severity=Severity(d["severity"]),
        condition_score=float(d["condition_score"]), needs_review=bool(d["needs_review"]),
        priority=Priority(d["priority"]), findings=[finding_from_dict(f) for f in d.get("findings", [])],
        lat=d.get("lat"), lon=d.get("lon"), location_source=d.get("location_source", "none"),
        confidence=float(d.get("confidence", 0.0)), n_observations=int(d.get("n_observations", 1)),
        best_frame_id=d.get("best_frame_id", ""),
        best_bbox=BBox.from_list(d["best_bbox"]) if d.get("best_bbox") else None,
        frame_ids=list(d.get("frame_ids", [])), crop_path=d.get("crop_path"), frame_path=d.get("frame_path"),
    )
