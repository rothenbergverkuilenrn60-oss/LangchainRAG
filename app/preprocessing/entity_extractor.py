"""
entity_extractor.py — Medical entity recognition module
Adopts a dual-layer recognition strategy:
  Layer 1: Rule/dictionary matching (high precision, zero latency) — recognizes drug names, diseases, symptoms, signs, etc.
  Layer 2: spaCy NER (general named entities, English) — recognizes person names, organizations, dates, etc.

Output format: {"疾病": [...], "药物": [...], "症状": [...], "体征": [...], ...}
These entities are stored in Chroma metadata, supporting precise entity-based filtered retrieval.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Medical dictionaries (rule layer)
# In production, should be loaded from UMLS / SNOMED-CT / DrugBank;
# core vocabulary is built in here.
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_VOCAB = {
    # Cardiovascular
    "冠心病", "心肌梗死", "心绞痛", "心力衰竭", "心房颤动", "心律失常",
    "高血压", "低血压", "主动脉夹层", "肺栓塞", "深静脉血栓",
    # Respiratory
    "肺炎", "哮喘", "慢性阻塞性肺疾病", "肺结核", "支气管炎", "胸膜炎",
    "肺癌", "急性呼吸窘迫综合征",
    # Digestive
    "胃炎", "胃溃疡", "十二指肠溃疡", "肝炎", "肝硬化", "胰腺炎",
    "克罗恩病", "溃疡性结肠炎", "肠梗阻", "阑尾炎",
    # Endocrine
    "糖尿病", "甲状腺功能亢进", "甲状腺功能减退", "库欣综合征",
    "艾迪生病", "高脂血症", "痛风",
    # Neurological
    "脑卒中", "脑梗死", "脑出血", "癫痫", "帕金森病", "阿尔茨海默病",
    "偏头痛", "脑膜炎", "吉兰-巴雷综合征",
    # Infectious
    "败血症", "脓毒血症", "艾滋病", "流感", "新冠肺炎",
    "疟疾", "伤寒", "布鲁氏菌病",
    # Renal
    "肾炎", "肾衰竭", "肾结石", "肾病综合征", "尿路感染",
    # Hematologic
    "贫血", "白血病", "淋巴瘤", "血小板减少症", "血友病",
    # Musculoskeletal
    "类风湿关节炎", "系统性红斑狼疮", "骨关节炎", "骨质疏松症",
    "强直性脊柱炎",
}

DRUG_VOCAB = {
    # Antibiotics
    "阿莫西林", "氨苄西林", "头孢菌素", "青霉素", "甲硝唑", "环丙沙星",
    "左氧氟沙星", "万古霉素", "利奈唑胺", "多西环素", "阿奇霉素",
    # Cardiovascular
    "阿司匹林", "氯吡格雷", "华法林", "肝素", "他汀类", "辛伐他汀",
    "阿托伐他汀", "美托洛尔", "阿替洛尔", "硝酸甘油", "硝苯地平",
    "氨氯地平", "依那普利", "卡托普利", "坎地沙坦", "呋塞米",
    # Analgesic/anesthetic
    "吗啡", "芬太尼", "对乙酰氨基酚", "布洛芬", "双氯芬酸", "曲马多",
    # Endocrine
    "胰岛素", "二甲双胍", "格列本脲", "左甲状腺素", "甲巯咪唑",
    # Neurological/psychiatric
    "氟西汀", "舍曲林", "奥氮平", "氯丙嗪", "苯妥英", "卡马西平",
    "丙戊酸", "地西泮", "艾司唑仑",
    # Hormones
    "泼尼松", "地塞米松", "甲泼尼龙", "氢化可的松",
    # Other
    "奥美拉唑", "兰索拉唑", "西替利嗪", "氯苯那敏",
}

SYMPTOM_VOCAB = {
    "发热", "咳嗽", "咳痰", "气短", "呼吸困难", "胸痛", "胸闷",
    "心悸", "头痛", "头晕", "恶心", "呕吐", "腹痛", "腹泻", "便秘",
    "乏力", "疲劳", "食欲不振", "体重下降", "水肿", "黄疸", "皮疹",
    "瘙痒", "失眠", "嗜睡", "意识障碍", "抽搐", "晕厥", "血尿",
    "蛋白尿", "尿频", "尿急", "尿痛", "关节痛", "肌痛", "背痛",
    "颈痛", "吞咽困难", "声音嘶哑", "鼻塞", "流涕", "喉痛",
}

SIGN_VOCAB = {
    "心动过速", "心动过缓", "高血压", "低血压", "发绀", "苍白",
    "肝大", "脾大", "淋巴结肿大", "杵状指", "颈静脉怒张",
    "腹水", "胸腔积液", "肺部啰音", "哮鸣音", "捻发音",
    "Babinski征", "Kernig征", "Brudzinski征",
}

LAB_VOCAB = {
    "白细胞", "红细胞", "血红蛋白", "血小板", "中性粒细胞",
    "血糖", "糖化血红蛋白", "肌酐", "尿素氮", "尿酸",
    "ALT", "AST", "胆红素", "白蛋白", "凝血酶原时间",
    "C反应蛋白", "降钙素原", "D-二聚体", "肌钙蛋白", "BNP",
    "钠", "钾", "氯", "钙", "磷", "镁",
    "血培养", "尿培养", "痰培养",
}

PROCEDURE_VOCAB = {
    "心电图", "超声心动图", "胸部X线", "CT", "MRI", "PET",
    "冠状动脉造影", "心导管", "内镜", "结肠镜", "胃镜",
    "腰椎穿刺", "骨髓穿刺", "活检", "手术",
    "透析", "化疗", "放疗", "免疫治疗",
}

# Merge all dictionaries for fast membership testing
ALL_VOCAB_BY_TYPE: Dict[str, set] = {
    "疾病": DISEASE_VOCAB,
    "药物": DRUG_VOCAB,
    "症状": SYMPTOM_VOCAB,
    "体征": SIGN_VOCAB,
    "实验室指标": LAB_VOCAB,
    "操作/检查": PROCEDURE_VOCAB,
}


# ─────────────────────────────────────────────────────────────────────────────
# Entity extractor
# ─────────────────────────────────────────────────────────────────────────────

class MedicalEntityExtractor:
    """
    Dual-layer medical entity recognizer.

    Usage:
        extractor = MedicalEntityExtractor()
        entities = extractor.extract(text)
        # → {"疾病": ["高血压", ...], "药物": [...], ...}
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy  # Enable English NER by default
        self._nlp = None  # Lazily load spaCy to avoid slow startup

        # Pre-compile dictionary lookup regex (alternative to Aho-Corasick)
        # Build regex per entity type, supports exact matching
        self._vocab_patterns: Dict[str, re.Pattern] = {}
        for entity_type, vocab in ALL_VOCAB_BY_TYPE.items():
            # Sort terms by length descending to ensure longer terms match first
            # (prevents "心肌" matching when "心" is a shorter term)
            sorted_terms = sorted(vocab, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_terms]
            if escaped:
                self._vocab_patterns[entity_type] = re.compile(
                    '(?:' + '|'.join(escaped) + ')'
                )

    def _load_spacy(self):
        """Lazily loads spaCy model"""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                # Use installed English model (handles English medical terminology)
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"spaCy loading failed, falling back to dictionary-only mode: {e}")
                self._nlp = None
                self.use_spacy = False

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts medical entities from text.

        Args:
            text: Input text (Chinese/English mixed)

        Returns:
            {entity_type: [entity_list]}, deduplicated
        """
        entities: Dict[str, List[str]] = {k: [] for k in ALL_VOCAB_BY_TYPE.keys()}

        # ── Layer 1: Dictionary regex matching ───────────────────────────────
        for entity_type, pattern in self._vocab_patterns.items():
            found = pattern.findall(text)
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for f in found:
                if f not in seen:
                    seen.add(f)
                    unique.append(f)
            entities[entity_type] = unique

        # ── Layer 2: spaCy NER (English entities) ────────────────────────────
        if self.use_spacy:
            self._load_spacy()
            if self._nlp:
                # Only process English parts (extract consecutive ASCII segments)
                english_segments = re.findall(r'[A-Za-z][A-Za-z0-9\s\-,\.]{10,}', text)
                for seg in english_segments[:5]:  # Limit segments processed to control latency
                    try:
                        doc = self._nlp(seg[:500])  # Limit length
                        for ent in doc.ents:
                            if ent.label_ in ("PERSON", "ORG", "GPE"):
                                # English proper nouns go into "other" category
                                if "其他" not in entities:
                                    entities["其他"] = []
                                if ent.text not in entities["其他"]:
                                    entities["其他"].append(ent.text)
                    except Exception:
                        pass

        # Remove empty lists to reduce metadata size
        return {k: v for k, v in entities.items() if v}

    def extract_from_query(self, query: str) -> Dict[str, List[str]]:
        """
        Extracts entities from a user query (same as extract, but additionally handles colloquial expressions).
        For example: maps "发烧" to the dictionary's "发热".
        """
        # Colloquial → standard term mapping
        COLLOQUIAL_MAP = {
            "发烧": "发热",
            "拉肚子": "腹泻",
            "胃疼": "腹痛",
            "肚子痛": "腹痛",
            "头疼": "头痛",
            "气喘": "呼吸困难",
            "血压高": "高血压",
            "血糖高": "糖尿病",
            "心脏病": "冠心病",
            "感冒": "流感",
        }
        normalized = query
        for colloquial, standard in COLLOQUIAL_MAP.items():
            normalized = normalized.replace(colloquial, standard)

        return self.extract(normalized)
