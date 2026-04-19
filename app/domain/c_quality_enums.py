from __future__ import annotations

from enum import Enum


class CoverageStatus(str, Enum):
    COVERED = "COVERED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"


class RequirementType(str, Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    LOGGING = "logging"
    STORAGE = "storage"
    INTERFACE = "interface"
    FUNCTIONAL = "functional"
    OTHER = "other"


class CoverageUnitType(str, Enum):
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TEST_STEP = "test_step"
    EXPECTED_RESULT = "expected_result"
    PRECONDITION = "precondition"
    TABLE_ROW_TEXT = "table_row_text"


class Modality(str, Enum):
    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"
    MAY = "may"
    UNKNOWN = "unknown"


class LLMLabel(str, Enum):
    COVERED = "COVERED"
    PARTIAL = "PARTIAL"
    CONFLICT = "CONFLICT"
    IRRELEVANT = "IRRELEVANT"
