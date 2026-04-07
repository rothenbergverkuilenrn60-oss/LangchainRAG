"""
rule_engine.py — Medical domain rule engine
Defines an extensible rule chain for medical documents like the Merck Manual,
processing text through:
  1. Drug name / disease name normalization (synonym unification)
  2. Measurement unit normalization (mg/kg → mg·kg⁻¹, etc.)
  3. Medical abbreviation expansion
  4. Table/formula content detection and marking
  5. Important medical warning recognition (black-box warnings, contraindications, etc.)
  6. Text quality scoring (filter overly short / garbled chunks)

Architecture pattern: Chain of Responsibility — each Rule is an independent processing step,
can be dynamically enabled/disabled for easy maintenance and extension.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rule base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseRule(ABC):    # ABC (Abstract Base Class): forces subclasses to implement apply(), otherwise not instantiable
    """
    All rules must implement this interface.
    apply() receives text and a metadata dict, returns (modified text, updated metadata).
    """
    name: str = "base_rule"
    enabled: bool = True

    @abstractmethod
    def apply(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Concrete rule implementations
# ─────────────────────────────────────────────────────────────────────────────

class QualityFilterRule(BaseRule):
    """
    Rule 1: Quality filter
    - Chunks shorter than MIN_CHARS characters are marked as low quality
    - Garbled character detection: marked if ratio of non-Chinese/English/digit chars exceeds threshold
    """
    name = "quality_filter"
    MIN_CHARS = 20                   # Minimum valid character count
    GARBAGE_CHAR_RATIO = 0.3         # Garbled character ratio threshold

    # Valid characters: Chinese, English letters, digits, common punctuation
    VALID_CHARS_RE = re.compile(r'[\u4e00-\u9fff\w\s，。？！、；：""''（）【】《》]')

    def apply(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # Compute valid character ratio
        valid_count = len(self.VALID_CHARS_RE.findall(text))
        total = max(len(text), 1)
        garbage_ratio = 1.0 - valid_count / total

        is_low_quality = (
            len(text.strip()) < self.MIN_CHARS
            or garbage_ratio > self.GARBAGE_CHAR_RATIO
        )
        # Write quality info into metadata (don't delete directly; let Indexer decide whether to skip)
        metadata["is_low_quality"] = is_low_quality
        metadata["garbage_ratio"] = round(garbage_ratio, 3)
        return text, metadata


class DrugNameNormalizationRule(BaseRule):
    """
    Rule 2: Drug name normalization
    Unifies common drug aliases/trade names to generic names (INN),
    preventing the same drug from being missed in retrieval.
    Sample entries listed here; production should load a complete drug dictionary (e.g., DrugBank).
    """
    name = "drug_normalization"

    # {alias/trade name: generic name}
    DRUG_SYNONYMS: Dict[str, str] = {
        "阿司匹林": "乙酰水杨酸",
        "百忧解": "氟西汀",
        "泰诺": "对乙酰氨基酚",
        "扑热息痛": "对乙酰氨基酚",
        "布洛芬": "异丁苯丙酸",  # Keep generic name
        "青霉素G": "苄青霉素",
        "维生素C": "抗坏血酸",
        "维生素B12": "氰钴胺",
        "肾上腺素": "肾上腺素",  # Already generic name, no change
        "扑尔敏": "氯苯那敏",
        "地塞米松": "地塞米松",
        "胰岛素": "胰岛素",
        "华法林": "华法林",
        "阿托品": "阿托品",
        "吗啡": "吗啡",
    }

    # Pre-compiled replacement regex: built once to avoid recompiling on every apply call
    _pattern: Optional[re.Pattern] = None

    def __init__(self):
        # Build all synonyms into a single OR regex for efficient replacement
        escaped = [re.escape(k) for k in self.DRUG_SYNONYMS.keys()]
        if escaped:
            self._pattern = re.compile('|'.join(escaped))

    def apply(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if self._pattern is None:
            return text, metadata

        def _replacer(m: re.Match) -> str:
            # Find the matched alias and return the corresponding generic name
            '''
            Original: "患者服用了泰诺后感觉好多了"
                           ↓
            _pattern scans the text, finds "泰诺" (index 7~9)
                                    ↓
            Engine creates re.Match object:
                m.group(0) = "泰诺"       ← matched content
                m.start()  = 7            ← start position
                m.end()    = 9            ← end position
                                    ↓
            Passes Match object to _replacer(m)
                                    ↓
            _replacer executes:
                self.DRUG_SYNONYMS.get("泰诺", "泰诺")
                → looks up dict, finds "泰诺" → "对乙酰氨基酚"
                → returns "对乙酰氨基酚"
                                    ↓
            Engine replaces "泰诺" in text with "对乙酰氨基酚"
                                    ↓
            Result: "患者服用了对乙酰氨基酚后感觉好多了"

            self.DRUG_SYNONYMS.get(m.group(0), m.group(0))
            #                  ↑ look up this term   ↑ return as-is if not found
            '''
            return self.DRUG_SYNONYMS.get(m.group(0), m.group(0))

        normalized = self._pattern.sub(_replacer, text)
        if normalized != text:
            metadata["drug_normalized"] = True  # Record that drug name normalization occurred
        return normalized, metadata


class MedicalAbbreviationRule(BaseRule):
    """
    Rule 3: Medical abbreviation expansion
    Inserts the full form after the first occurrence of an abbreviation
    to improve LLM comprehension.
    """
    name = "abbreviation_expansion"

    ABBREVIATIONS: Dict[str, str] = {
        "HTN": "高血压(HTN)",
        "DM": "糖尿病(DM)",
        "DM2": "2型糖尿病(DM2)",
        "CAD": "冠状动脉疾病(CAD)",
        "MI": "心肌梗死(MI)",
        "CHF": "充血性心力衰竭(CHF)",
        "COPD": "慢性阻塞性肺疾病(COPD)",
        "CKD": "慢性肾脏病(CKD)",
        "DVT": "深静脉血栓(DVT)",
        "PE": "肺栓塞(PE)",
        "CVA": "脑血管意外(CVA)",
        "TIA": "短暂性脑缺血发作(TIA)",
        "ECG": "心电图(ECG)",
        "MRI": "磁共振成像(MRI)",
        "CT": "计算机断层扫描(CT)",
        "IV": "静脉注射(IV)",
        "IM": "肌肉注射(IM)",
        "PO": "口服(PO)",
        "PRN": "按需(PRN)",
        "bid": "每日两次(bid)",
        "tid": "每日三次(tid)",
        "qid": "每日四次(qid)",
        "qd": "每日一次(qd)",
        "WBC": "白细胞(WBC)",
        "RBC": "红细胞(RBC)",
        "Hb": "血红蛋白(Hb)",
        "BP": "血压(BP)",
        "HR": "心率(HR)",
    }

    def apply(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        expanded = text
        found_abbrevs = []
        for abbr, full in self.ABBREVIATIONS.items():
            # Only replace standalone abbreviations (non-letter boundaries), using \b word boundary
            pattern = r'\b' + re.escape(abbr) + r'\b'   # Escape abbreviation as regex-safe literal
            # Only replace the first occurrence (expand once, keep original form thereafter)
            new_text, count = re.subn(pattern, full, expanded, count=1)
            if count > 0:
                found_abbrevs.append(abbr)
                expanded = new_text
        if found_abbrevs:
            metadata["expanded_abbreviations"] = found_abbrevs
        return expanded, metadata


class TableDetectionRule(BaseRule):
    """
    Rule 4: Table/image content detection
    Tables extracted from PDFs often appear as columns of numbers separated by many spaces.
    Marks detected tables in metadata for downstream Chunker to decide on separate handling.
    """
    name = "table_detection"

    # Numeric columns with consecutive spaces: e.g., "45   67   89   120"
    TABLE_ROW_RE = re.compile(r'(\d+[\.,]?\d*\s{2,}){3,}')
    # Consecutive Chinese+space columns (table headers)
    TABLE_HEADER_RE = re.compile(r'([\u4e00-\u9fff]{1,8}\s{2,}){3,}')

    def apply(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        has_table = bool(
            self.TABLE_ROW_RE.search(text)
            or self.TABLE_HEADER_RE.search(text)
        )
        metadata["has_table"] = has_table
        if has_table:
            # Prefix table chunks to help LLM recognize structured content
            text = "[TABLE CONTENT]\n" + text
        return text, metadata


class WarningDetectionRule(BaseRule):
    """
    Rule 5: Medical warning recognition
    Identifies black-box warnings, contraindications, severe adverse reactions, and other high-risk content.
    Marks in metadata so retrieval can boost their weight.
    """
    name = "warning_detection"

    WARNING_PATTERNS = [
        re.compile(r'(警告|黑框警告|BLACK BOX WARNING)', re.IGNORECASE),    # re.IGNORECASE: case-insensitive matching
        re.compile(r'(禁忌|禁忌症|contraindication)', re.IGNORECASE),
        re.compile(r'(严重不良反应|严重副作用|致死|fatal|life.threatening)', re.IGNORECASE),
        re.compile(r'(过量|中毒|overdose|toxicity)', re.IGNORECASE),
        re.compile(r'(慎用|不宜|不应|避免使用)', re.IGNORECASE),
    ]

    def apply(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        warning_types = []
        for pattern in self.WARNING_PATTERNS:
            if pattern.search(text):
                warning_types.append(pattern.pattern)

        metadata["has_warning"] = bool(warning_types)
        metadata["warning_types"] = warning_types
        # Chunks with warnings should receive higher weight in retrieval ranking
        metadata["importance_boost"] = 1.5 if warning_types else 1.0
        return text, metadata


class UnitNormalizationRule(BaseRule):
    """
    Rule 6: Measurement unit normalization
    Unifies dose, concentration, and other unit formats to reduce semantic dispersion in retrieval.
    """
    name = "unit_normalization"

    # Common unit variants → standard forms: each (pattern, replacement) pair
    UNIT_MAP = [
        (re.compile(r'mg/kg'), 'mg·kg⁻¹'),
        (re.compile(r'mg/dl', re.IGNORECASE), 'mg/dL'),
        (re.compile(r'ml(?!\w)', re.IGNORECASE), 'mL'),  # (?!\w) negative lookahead: ensures ml isn't followed by alphanumeric
        (re.compile(r'ug(?!\w)', re.IGNORECASE), 'μg'),
        (re.compile(r'mcg(?!\w)', re.IGNORECASE), 'μg'),
        (re.compile(r'mmhg', re.IGNORECASE), 'mmHg'),
        (re.compile(r'u/l(?!\w)', re.IGNORECASE), 'U/L'),
    ]

    def apply(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        for pattern, replacement in self.UNIT_MAP:
            text = pattern.sub(replacement, text)   # Replace all pattern matches with replacement
        return text, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Rule engine (Chain of Responsibility orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class MedicalRuleEngine:
    """
    Medical text rule engine: executes a rule chain in order, supports dynamic enable/disable.

    Usage:
        engine = MedicalRuleEngine()
        cleaned_text, metadata = engine.process(raw_text, initial_metadata)
    """

    def __init__(self, custom_rules: Optional[List[BaseRule]] = None):
        # Default rule chain: ordered by recommended execution sequence
        # quality filter → unit normalization → drug name normalization → abbreviation expansion → table detection → warning recognition
        self._rules: List[BaseRule] = [
            QualityFilterRule(),
            UnitNormalizationRule(),
            DrugNameNormalizationRule(),
            MedicalAbbreviationRule(),
            TableDetectionRule(),
            WarningDetectionRule(),
        ]
        # Supports injection of custom rules, appended to chain end
        if custom_rules:
            self._rules.extend(custom_rules)

    def process(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Applies all enabled rules to text in sequence.

        Args:
            text: Input text
            metadata: Initial metadata dict (modified/extended in-place by rules)

        Returns:
            (processed_text, updated_metadata)
        """
        if metadata is None:
            metadata = {}

        for rule in self._rules:
            if not rule.enabled:
                continue  # Skip disabled rules
            try:
                text, metadata = rule.apply(text, metadata)
            except Exception as e:
                # Single rule failure doesn't affect overall flow; log warning and continue
                logger.warning(f"Rule [{rule.name}] execution error (skipped): {e}")

        return text, metadata

    def disable_rule(self, rule_name: str):
        """Disables a rule by name"""
        for rule in self._rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Rule [{rule_name}] disabled")
                return
        logger.warning(f"Rule not found: {rule_name}")

    def enable_rule(self, rule_name: str):
        """Enables a rule by name"""
        for rule in self._rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Rule [{rule_name}] enabled")
                return

    def list_rules(self) -> List[Dict[str, Any]]:
        """Lists all rules and their status"""
        return [
            {"name": r.name, "enabled": r.enabled, "class": type(r).__name__}
            for r in self._rules
        ]
