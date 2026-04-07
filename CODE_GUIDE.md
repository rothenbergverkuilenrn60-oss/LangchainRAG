# Medical RAG 系统代码讲解文档

> 本文档对系统每个文件的每一行关键代码进行功能说明，并在最后提供完整的数据流讲解。

---

## 目录

1. [项目总体结构](#1-项目总体结构)
2. [app/config.py — 全局配置](#2-appconfigpy--全局配置)
3. [app/models.py — 数据模型](#3-appmodelspy--数据模型)
4. [app/preprocessing/pdf_processor.py — PDF预处理](#4-apppreprocessingpdf_processorpy--pdf预处理)
5. [app/preprocessing/rule_engine.py — 规则引擎](#5-apppreprocessingrule_enginepy--规则引擎)
6. [app/preprocessing/entity_extractor.py — 实体识别](#6-apppreprocessingentity_extractorpy--实体识别)
7. [app/indexing/chunker.py — 文档切分](#7-appindexingchunkerpy--文档切分)
8. [app/indexing/indexer.py — 向量索引](#8-appindexingindexerpy--向量索引)
9. [app/retrieval/query_processor.py — 查询处理](#9-appretrievalquery_processorpy--查询处理)
10. [app/retrieval/rrf.py — 互惠排名融合](#10-appretrievalrrfpy--互惠排名融合)
11. [app/retrieval/hybrid_retriever.py — 混合检索](#11-appretrievalhybrid_retrieverpy--混合检索)
12. [app/retrieval/reranker.py — 重排序](#12-appretrievalrerankerpy--重排序)
13. [app/generation/context_processor.py — 上下文处理](#13-appgenerationcontext_processorpy--上下文处理)
14. [app/generation/generator.py — LLM生成](#14-appgenerationgeneratorpy--llm生成)
15. [app/tenants/manager.py — 多租户管理](#15-apptenantsmanagerpy--多租户管理)
16. [app/pipeline.py — 主流水线](#16-apppipelinepy--主流水线)
17. [app/main.py — FastAPI入口](#17-appmainpy--fastapi入口)
18. [scripts/ingest.py — 摄入脚本](#18-scriptsingestpy--摄入脚本)
19. [完整数据流讲解](#19-完整数据流讲解)

---

## 1. 项目总体结构

```
Langchain/
├── app/
│   ├── config.py              # 全局配置（环境变量读取）
│   ├── models.py              # Pydantic 数据模型（API 输入/输出结构）
│   ├── pipeline.py            # 主流水线（串联所有模块）
│   ├── main.py                # FastAPI HTTP 接口定义
│   ├── preprocessing/
│   │   ├── pdf_processor.py   # PDF 文本提取与清洗
│   │   ├── rule_engine.py     # 医学领域规则引擎（责任链模式）
│   │   └── entity_extractor.py# 医学实体识别（双层策略）
│   ├── indexing/
│   │   ├── chunker.py         # 语义分块（父子块 + 语义边界检测）
│   │   └── indexer.py         # ChromaDB + BM25 索引管理
│   ├── retrieval/
│   │   ├── query_processor.py # 查询扩展（重写/HyDE/Multi-Query）
│   │   ├── rrf.py             # 双层 RRF 融合 + 去重
│   │   ├── hybrid_retriever.py# 混合检索协调器
│   │   └── reranker.py        # BGE-Reranker 精排
│   ├── generation/
│   │   ├── context_processor.py# 上下文压缩 + 长上下文重排
│   │   └── generator.py       # 流式 LLM 生成 + Self-RAG
│   └── tenants/
│       └── manager.py         # 多租户生命周期管理
├── scripts/
│   └── ingest.py              # 命令行文档摄入工具
├── floder/
│   └── 默克诊疗手册.pdf        # 源文档
└── .env                       # 环境变量（API Key 等）
```

**设计原则**：
- **单向依赖**：`main → pipeline → retrieval/generation → indexing → preprocessing`，上层模块不被下层反向依赖
- **延迟加载**：耗时组件（模型）在第一次请求时才加载，加快启动速度
- **异步优先**：所有 I/O 密集操作（网络请求、向量计算）使用 async/await

---

## 2. app/config.py — 全局配置

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
```
- `pydantic_settings`：Pydantic v2 的扩展库，专门用于从环境变量读取配置
- `BaseSettings`：继承它的类会自动从环境变量 + .env 文件读取字段值

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",           # 告诉 pydantic-settings 去哪里读取 .env
        env_file_encoding="utf-8", # .env 文件编码（中文路径需要 utf-8）
        extra="ignore",            # 忽略 .env 中多余的键，避免报错
    )
```
- `SettingsConfigDict`：配置类的元数据，控制 .env 加载行为
- `extra="ignore"`：如果 .env 有项目不认识的键（如旧配置），不报错直接忽略

```python
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
```
- 字段名（小写+下划线）自动对应环境变量名（DEEPSEEK_API_KEY）
- 右侧是默认值，如果环境变量未设置则使用默认值

```python
    embedding_model_path: str = "/mnt/f/my_models/..."
    reranker_model_path: str = "/mnt/f/my_models/..."
```
- 本地模型路径，Windows 路径挂载到 WSL 的 /mnt/f 下
- 不需要下载，直接加载本地离线模型

```python
    parent_chunk_size: int = 1000
    child_chunk_size: int = 300
    semantic_breakpoint_threshold: float = 0.45
```
- 分块超参数：父块 1000 字符足够承载完整段落，子块 300 字符精确匹配
- `0.45` 阈值：相邻句子余弦相似度 < 0.45 时认为是语义边界（经验值）

```python
settings = Settings()  # 模块级单例
```
- 在模块导入时立即实例化，其他模块通过 `from app.config import settings` 使用
- 整个进程只有一份 Settings 实例（单例模式）

---

## 3. app/models.py — 数据模型

```python
from pydantic import BaseModel, Field
```
- `BaseModel`：所有数据模型的基类，提供类型验证、序列化、文档生成
- `Field`：为字段添加元数据（描述、验证规则、默认值）

```python
class QueryMode(str, Enum):
    full = "full"
    rewrite_only = "rewrite_only"
    basic = "basic"
```
- 继承 `str` 使枚举值直接序列化为字符串（JSON 兼容）
- `full`：启用所有查询扩展策略（最高召回，最慢）
- `basic`：只用原始查询（最快，适合简单问题）

```python
class DocumentChunk(BaseModel):
    chunk_id: str                     # UUID，全局唯一
    text: str                         # 块的实际文本内容
    parent_chunk_id: Optional[str]    # None = 自己是父块
    page_numbers: List[int] = []      # 可能跨多页
    entities: Dict[str, List[str]] = {}  # {"疾病": ["高血压"], "药物": [...]}
```
- 这是流水线内部传递的核心数据结构
- `parent_chunk_id`：实现父子关联，检索子块后能找回父块上下文

```python
class RetrievedChunk(BaseModel):
    rrf_score: float = 0.0      # 双层 RRF 融合后的分数
    rerank_score: float = 0.0   # BGE-Reranker 精确打分（最终排序依据）
```
- 保留两种分数便于调试（可以看 RRF 认为相关但 Reranker 降权的情况）

```python
class CitationSource(BaseModel):
    citation_id: int    # 对应正文中的 [来源 1], [来源 2] 编号
    excerpt: str        # 引用原文摘录（前 200 字符展示给用户）
```
- 细粒度溯源的核心结构，每条引用都能追溯到具体页码和章节

```python
class StreamEvent(BaseModel):
    event: str    # "token" | "citations" | "metadata" | "done" | "error"
    data: Any     # 根据 event 类型不同，data 结构不同
```
- SSE（Server-Sent Events）流中每个事件的格式
- 前端根据 `event` 字段判断如何处理 `data`

---

## 4. app/preprocessing/pdf_processor.py — PDF预处理

```python
from pypdf import PdfReader
```
- `pypdf` 是 PyPDF2 的继任者，对中文 PDF 支持更好
- `PdfReader` 逐页提取文本，保留页面顺序

```python
CHAPTER_RE = re.compile(
    r'^(第\s*[一二三四五六七八九十百\d]+\s*[篇章部])\s+(.+)$',
    re.MULTILINE,
)
```
- 正则匹配中文章节标题格式：`第XX篇 标题名`
- `re.MULTILINE`：`^` 和 `$` 匹配每一行的行首/行尾（不只是整个字符串的首尾）
- `\s*` 允许"第 1 篇"（有空格）和"第1篇"（无空格）两种写法

```python
PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
```
- 匹配孤立的页码行（只有 1-4 位数字的行，如 `   123   `）
- PDF 提取时页码通常单独一行，需要删除避免干扰文本

```python
reader = PdfReader(str(self.pdf_path))
total_pages = len(reader.pages)
```
- `len(reader.pages)` 获取总页数（默克诊疗手册约 1000+ 页）
- `str()` 转换：PdfReader 需要字符串路径，不接受 Path 对象

```python
for page_idx, page in enumerate(reader.pages):
    raw_text = page.extract_text() or ""
```
- `enumerate` 同时获取索引和页面对象
- `or ""` 处理空页面（`extract_text()` 可能返回 `None`）

```python
    cleaned = self._clean_text(raw_text)
    chapter_match = CHAPTER_RE.search(cleaned)
    if chapter_match:
        current_chapter = f"{chapter_match.group(1)} {chapter_match.group(2)}".strip()
        current_section = ""  # 进入新章，清空节标题
```
- 使用跨页状态变量 `current_chapter` 追踪章节，因为章节标题只在某一页出现
- `group(1)` 是第一个捕获组（"第X篇"），`group(2)` 是章节名

```python
def _clean_text(self, raw: str) -> str:
    text = PAGE_NUM_RE.sub('', raw)      # 删除页码行
    # 过滤短行中含书名/版权的页眉页脚
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 30 and HEADER_FOOTER_RE.search(stripped):
            continue   # 跳过：短行且含页眉特征词
        filtered.append(line)
```
- 两步过滤：先删孤立页码，再逐行检测页眉页脚
- 只过滤 `< 30` 字符的短行，避免误删正文中碰巧含"默克"字样的长句

```python
    text = SOFT_HYPHEN_RE.sub('', text)  # "处-\n理" → "处理"
    text = MULTI_BLANK_RE.sub('\n\n', text)  # 多空行 → 单空行
```
- PDF 换行时会插入连字符，需要还原被分割的词
- 压缩多余空行让文本更紧凑，减少 token 消耗

---

## 5. app/preprocessing/rule_engine.py — 规则引擎

**设计模式：责任链（Chain of Responsibility）**
每个 Rule 是一个独立处理步骤，引擎按顺序调用，每个 Rule 只处理自己关心的情况。

```python
class BaseRule(ABC):
    @abstractmethod
    def apply(self, text: str, metadata: Dict) -> Tuple[str, Dict]:
        ...
```
- `ABC`（Abstract Base Class）：强制子类实现 `apply` 方法，否则无法实例化
- 输入输出都是 `(text, metadata)`，便于链式传递

```python
class QualityFilterRule(BaseRule):
    MIN_CHARS = 20
    GARBAGE_CHAR_RATIO = 0.3

    VALID_CHARS_RE = re.compile(r'[\u4e00-\u9fff\w\s，。？！、；：""''（）【】《》]')

    def apply(self, text, metadata):
        valid_count = len(self.VALID_CHARS_RE.findall(text))
        total = max(len(text), 1)
        garbage_ratio = 1.0 - valid_count / total
        is_low_quality = len(text.strip()) < self.MIN_CHARS or garbage_ratio > self.GARBAGE_CHAR_RATIO
        metadata["is_low_quality"] = is_low_quality
```
- `\u4e00-\u9fff`：Unicode 中文基本汉字区间（CJK Unified Ideographs）
- 乱码比例 = 1 - 有效字符比例，超过 30% 标记为低质量
- **不直接删除**，只标记，由 Indexer 决定是否跳过（关注点分离）

```python
class DrugNameNormalizationRule(BaseRule):
    DRUG_SYNONYMS = {"百忧解": "氟西汀", "泰诺": "对乙酰氨基酚", ...}

    def __init__(self):
        escaped = [re.escape(k) for k in self.DRUG_SYNONYMS.keys()]
        self._pattern = re.compile('|'.join(escaped))
```
- 将所有同义词构建为一个 OR 正则 `百忧解|泰诺|...`
- 一次正则扫描完成所有替换，比循环逐一替换快 N 倍（N = 词典大小）
- `re.escape()` 转义特殊字符（如括号），避免正则解析错误

```python
    def apply(self, text, metadata):
        def _replacer(m: re.Match) -> str:
            return self.DRUG_SYNONYMS.get(m.group(0), m.group(0))
        normalized = self._pattern.sub(_replacer, text)
```
- `_replacer` 是回调函数，在每次匹配时调用
- `m.group(0)` 是完整匹配串（别名），查字典返回通用名

```python
class TableDetectionRule(BaseRule):
    TABLE_ROW_RE = re.compile(r'(\d+[\.,]?\d*\s{2,}){3,}')
```
- 匹配"连续多组：数字后跟2+个空格"的模式（典型表格行）
- `{3,}` 要求至少 3 组，避免把普通数字句误判为表格

```python
class MedicalRuleEngine:
    def __init__(self):
        self._rules = [
            QualityFilterRule(),    # 先过滤再处理，避免浪费计算
            UnitNormalizationRule(),
            DrugNameNormalizationRule(),
            MedicalAbbreviationRule(),
            TableDetectionRule(),
            WarningDetectionRule(),
        ]
```
- 顺序设计：质量过滤放最前，低质量块后续规则也不需要跑了

```python
    def process(self, text, metadata=None):
        for rule in self._rules:
            if not rule.enabled:
                continue
            try:
                text, metadata = rule.apply(text, metadata)
            except Exception as e:
                logger.warning(f"规则 [{rule.name}] 执行出错（已跳过）: {e}")
        return text, metadata
```
- `try/except` 保证单规则失败不影响整个流程（容错设计）
- `rule.enabled` 允许运行时动态禁用规则（如线上发现某规则有 bug）

---

## 6. app/preprocessing/entity_extractor.py — 实体识别

**双层策略**：Layer1 词典匹配（快、中文友好） + Layer2 spaCy NER（英文实体）

```python
DISEASE_VOCAB = {"冠心病", "心肌梗死", "高血压", ...}  # Python set，O(1) 成员检测
```
- 使用 `set` 而不是 `list`：成员检测 O(1) vs O(n)，词典越大越明显
- 按系统分类组织：心血管、呼吸、消化、内分泌...便于维护和扩展

```python
    def __init__(self, use_spacy=True):
        for entity_type, vocab in ALL_VOCAB_BY_TYPE.items():
            sorted_terms = sorted(vocab, key=len, reverse=True)  # 长词优先
            escaped = [re.escape(t) for t in sorted_terms]
            self._vocab_patterns[entity_type] = re.compile(
                '(?:' + '|'.join(escaped) + ')'
            )
```
- **长词优先**的关键：防止"心肌梗死"被"心肌"提前匹配（子串覆盖问题）
- `(?:...)`：非捕获分组，比 `(...)` 更快（不需要保存分组内容）

```python
    def extract(self, text):
        for entity_type, pattern in self._vocab_patterns.items():
            found = pattern.findall(text)
            seen = set()
            unique = []
            for f in found:
                if f not in seen:
                    seen.add(f)
                    unique.append(f)
```
- `seen` set 去重的同时用 `unique` list 保持首次出现的顺序
- 比 `list(set(found))` 好：set 无序，可能打乱提取顺序

```python
        if self.use_spacy:
            english_segments = re.findall(r'[A-Za-z][A-Za-z0-9\s\-,\.]{10,}', text)
            for seg in english_segments[:5]:  # 限制 5 段
                doc = self._nlp(seg[:500])    # 限制 500 字符
                for ent in doc.ents:
                    if ent.label_ in ("PERSON", "ORG", "GPE"):
                        ...
```
- 只处理英文段落（中文由词典层处理），避免 spaCy 在中文上的乱识别
- `[:5]` 和 `[:500]` 双重限制，控制 spaCy 推理延迟在毫秒级

---

## 7. app/indexing/chunker.py — 文档切分

### SemanticChunker（语义分块器）

```python
SENT_SPLIT_RE = re.compile(r'(?<=[。！？；\.\!\?;])\s*')
```
- `(?<=...)` 是正则的"零宽后行断言"（lookbehind）：匹配位置，不消耗字符
- 在中文句号/问号/感叹号/分号**之后**切分，保留标点在句末

```python
    def split(self, text):
        sentences = [s.strip() for s in self.SENT_SPLIT_RE.split(text) if s.strip()]

        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            normalize_embeddings=True,  # L2 归一化后，余弦相似度 = 点积（更快）
        )
```
- 批量 encode 一次性计算所有句子的 embedding
- `normalize_embeddings=True`：归一化后点积即为余弦相似度，避免额外除法

```python
        for i in range(len(sentences) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.threshold:
                breakpoints.append(i + 1)  # 在 i+1 句开始新块
```
- 滑动窗口检测相邻句子的语义相似度
- 相似度 < 0.45（阈值）= 话题切换 = 语义边界 = 切块点

```python
def _cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0
```
- `np.dot`：向量点积（O(n) 时间，n 为向量维度 1024）
- `np.linalg.norm`：L2 范数（向量长度）
- 防零除：`if norm > 0` 处理零向量情况

### ParentChildChunker（父子块切分器）

```python
    def create_chunks(self, pages, tenant_id, doc_source):
        accumulated = ""
        accumulated_pages = []

        for text, page_num, chapter, section in full_text_parts:
            accumulated += text + "\n"
            accumulated_pages.append(page_num)

            if len(accumulated) >= settings.parent_chunk_size:
                parent_subs = self.semantic_chunker.split(accumulated)
```
- **跨页积累**：将多页文本合并到 `accumulated`，达到父块大小时统一切分
- 避免强制在页面边界切分（同一段落可能跨两页）

```python
    def _make_child_chunks(self, parent):
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            child_text = text[start:end]
            ...
            start += size - overlap  # 滑动步长 = 块大小 - 重叠区
```
- 固定大小 + 重叠的滑动窗口
- 重叠区确保跨块边界的信息不丢失（如"前半句在块A，后半句在块B"）
- `parent_chunk_id = parent.chunk_id`：子块记录父块 ID，形成双向关联

---

## 8. app/indexing/indexer.py — 向量索引

```python
self.client = chromadb.PersistentClient(
    path=settings.chroma_db_path,
    settings=ChromaSettings(
        anonymized_telemetry=False,  # 关闭匿名数据上报（隐私保护）
        allow_reset=True,
    ),
)
```
- `PersistentClient`：数据持久化到磁盘，进程重启后数据不丢失
- `anonymized_telemetry=False`：关闭 Chroma 的使用统计上报

```python
def _col_children(self, tenant_id):
    return f"{settings.collection_prefix}_{tenant_id}_children"
```
- 命名规范：`merck_rag_default_children`、`merck_rag_hospital_a_children`
- 不同租户使用不同集合名，实现数据隔离

```python
def _serialize_metadata(meta):
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            serialized[k] = v
        elif isinstance(v, (list, dict)):
            serialized[k] = json.dumps(v, ensure_ascii=False)
```
- Chroma 元数据值类型限制：只支持 `str/int/float/bool`
- 列表和字典需要序列化为 JSON 字符串，检索后反序列化

```python
def _batch_upsert(self, collection, chunks, batch_size):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = self.emb_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
        ).tolist()           # numpy array → Python list（Chroma 需要 list）
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
```
- `upsert = update + insert`：ID 已存在则更新，不存在则插入（幂等操作）
- `.tolist()`：将 numpy float32 数组转为 Python 原生 list（JSON 序列化兼容）

```python
def _build_bm25_index(self, chunks, tenant_id):
    tokenized = [list(jieba.cut(t)) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(index_path, "wb") as f:
        pickle.dump((bm25, chunk_ids, texts, metadatas_list), f)
```
- `jieba.cut`：中文精确分词（"高血压患者" → ["高血压", "患者"]）
- `BM25Okapi`：BM25 的经典变体，内置参数 k1=1.5, b=0.75（Robertson 推荐值）
- `pickle.dump`：序列化 BM25 对象到磁盘，避免每次启动重建（耗时约 3-5 分钟）

```python
def bm25_search(self, query_text, tenant_id, top_k=None):
    query_tokens = list(jieba.cut(query_text))  # 查询同样需要分词
    scores = bm25.get_scores(query_tokens)       # 对所有文档计算 BM25 分数
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]
```
- `get_scores` 计算查询词对所有文档的 BM25 分数（词频 × IDF 加权）
- `sorted` + `key=lambda` 按分数降序取 Top-K

---

## 9. app/retrieval/query_processor.py — 查询处理

```python
self._llm = ChatDeepSeek(
    temperature=0.3,   # 查询变体需要多样性，温度比生成（0.1）稍高
    max_tokens=512,    # 查询不需要长输出
)
```
- 查询扩展用温度 0.3：既有一定随机性（产生多样变体），又不会太随机
- 生成答案用温度 0.1：尽量确定性，减少幻觉

```python
async def expand(self, question, mode=QueryMode.full):
    tasks = []
    task_names = []
    tasks.append(self._rewrite_query(question))     # 协程1
    task_names.append("rewritten")
    if mode == QueryMode.full:
        tasks.append(self._generate_hyde(question))  # 协程2
        tasks.append(self._generate_multi_queries(question))  # 协程3

    results = await asyncio.gather(*tasks, return_exceptions=True)
```
- `asyncio.gather(*tasks)`：并发执行所有协程，总等待时间 ≈ 最慢任务的时间
- `return_exceptions=True`：某个任务失败不影响其他任务，返回异常对象而非抛出

```python
async def _generate_hyde(self, question):
    messages = [
        SystemMessage(content=HYDE_SYSTEM),   # 角色：临床医学专家
        HumanMessage(content=question),
    ]
    response = await self._llm.ainvoke(messages)
    hyde_doc = self._parser.invoke(response).strip()
```
- **HyDE 原理**：LLM 先"假设"一个理想答案，再用答案的 embedding 检索
- 答案的语义空间比问题更接近文档，检索命中率更高

---

## 10. app/retrieval/rrf.py — 互惠排名融合

**RRF 公式**：`score(d) = Σᵢ 1/(k + rankᵢ(d))`

- `k=60`：平滑常数，防止排名第1的文档分数过高（稀释第一名的绝对优势）
- 在所有排名列表中都出现的文档累积更高 RRF 分（多路互相印证）

```python
def rrf_merge(ranked_lists, k=None, top_n=None):
    doc_scores = defaultdict(lambda: {
        "doc": None, "rrf_score": 0.0, "appearances": 0, "ranks": []
    })

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, doc in enumerate(ranked_list):
            doc_key = _get_doc_key(doc)
            rrf_contribution = 1.0 / (k + rank + 1)   # rank 从 0 开始，+1 变成从 1 开始
            doc_scores[doc_key]["rrf_score"] += rrf_contribution
```
- `defaultdict(lambda: {...})`：键不存在时自动初始化，避免 KeyError
- `rank + 1`：RRF 公式的 rank 从 1 计数（排名第1的贡献 = 1/(60+1) ≈ 0.0164）

```python
def _get_doc_key(doc):
    chunk_id = meta.get("chunk_id", "")
    if chunk_id:
        return chunk_id
    text_snippet = doc.get("text", "")[:100]
    return hashlib.md5(text_snippet.encode()).hexdigest()
```
- 优先用 chunk_id 作唯一键（精确匹配）
- 回退用文本哈希（防止不同来源的同一文档被计两次）

```python
class DualRRFFusion:
    def fuse(self, query_results, top_n=None):
        # Layer 1：per-query，BM25+向量 → 混合排名
        for variant_name, retrieval_dict in query_results.items():
            bm25_docs = retrieval_dict.get("bm25", [])
            vector_docs = retrieval_dict.get("vector", [])
            hybrid = rrf_merge(ranked_lists=[bm25_docs, vector_docs], k=self.k)

        # Layer 2：cross-query，所有变体 → 最终候选
        layer2_merged = rrf_merge(ranked_lists=layer1_results, k=self.k)

        # 去重
        deduplicated = deduplicate(layer2_merged)
```
- **Layer1**：解决 BM25 和向量检索分数量纲不同的问题（RRF 只用排名）
- **Layer2**：解决不同查询变体检索结果的聚合问题（多变体互相印证）
- **去重**：同一 chunk 可能被多个查询变体都检索到，只保留一份

---

## 11. app/retrieval/hybrid_retriever.py — 混合检索

```python
async def retrieve(self, query_variants, tenant_id, filters=None, top_n=None):
    tasks = {
        variant_name: self._retrieve_single(
            query_text=query_text,
            variant_name=variant_name,
            tenant_id=tenant_id,
            filters=filters,
        )
        for variant_name, query_text in query_variants.items()
        if query_text  # 跳过空查询（HyDE 可能生成空串）
    }
    results_list = await asyncio.gather(*coros, return_exceptions=True)
```
- 字典推导式创建协程字典
- `asyncio.gather` 并发执行所有查询变体的检索，总延迟 ≈ 最慢变体的延迟

```python
async def _retrieve_single(self, query_text, ...):
    bm25_task = asyncio.to_thread(
        self.indexer.bm25_search, query_text=query_text, ...
    )
    vector_task = asyncio.to_thread(
        self.indexer.vector_search, query_text=query_text, ...
    )
    bm25_docs, vector_docs = await asyncio.gather(bm25_task, vector_task)
```
- `asyncio.to_thread`：将同步的 CPU/IO 密集操作放到线程池执行，不阻塞事件循环
- BM25 和向量检索也并发执行（两路检索同时进行）

```python
async def _expand_to_parents(self, child_docs, tenant_id):
    for doc in child_docs:
        parent_id = meta.get("parent_chunk_id", "")
        if parent_id:
            parent = await asyncio.to_thread(
                self.indexer.get_parent_by_id, parent_id=parent_id, ...
            )
            if parent:
                expanded_doc = {
                    **doc,
                    "text": parent["text"],      # 替换为父块全文
                    "child_text": doc["text"],   # 保留子块原文
```
- **Parent-Child 上下文扩展**：检索到子块后，拉取其父块全文送给 LLM
- 父块提供更完整的上下文，减少因切块导致的信息截断
- `**doc`：解包原有 doc 的所有字段，再覆盖 `text`（保留分数等元数据）

---

## 12. app/retrieval/reranker.py — 重排序

```python
self._model = CrossEncoder(
    model_name=self._model_path,
    device="cuda",
    max_length=512,   # query + document 总长度限制（超长截断）
)
```
- `CrossEncoder` vs `SentenceTransformer`：
  - CrossEncoder：query+doc 拼接输入，精度高，不能预计算
  - SentenceTransformer：分别编码，速度快，适合粗检索
- `max_length=512`：BGE-Reranker 的最大输入 token 数（约等于 300-400 中文字）

```python
    pairs = [(query, doc["text"]) for doc in docs]
    scores = self._model.predict(
        pairs,
        batch_size=32,
        convert_to_numpy=True,  # 返回 numpy array，比 tensor 更方便排序
    )
```
- CrossEncoder 输入格式：`[(query, doc1), (query, doc2), ...]`
- `predict` 返回每对的相关性 logit 值（无界，通常在 -10 到 10 之间）

```python
    scored_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )
```
- `zip(scores, docs)` 将分数和文档配对
- `sorted` 按分数降序排列
- 结果：相关性最高的文档排第一

---

## 13. app/generation/context_processor.py — 上下文处理

```python
RELEVANCE_THRESHOLD = -5.0  # BGE-Reranker logit 阈值
```
- BGE-Reranker-v2-M3 输出 logit 值（未经 sigmoid 的原始分数）
- 大于 0：相关；小于 0：不相关；-5.0 是经验性的低相关阈值

```python
def _reorder_for_llm(self, chunks):
    front_group = chunks[::2]   # 索引 0, 2, 4, ... → 最相关放前面
    back_group = chunks[1::2]   # 索引 1, 3, 5, ...
    reordered = front_group + list(reversed(back_group))
```
- **Lost in the Middle 缓解**：研究表明 LLM 对 prompt 中间部分注意力最低
- 把最相关的放在开头（绝对注意力最高）和结尾（相对注意力较高）
- `chunks[::2]`：Python 切片步长 2，取偶数索引

```python
def _build_citations(self, chunks):
    for i, chunk in enumerate(chunks):
        citation = CitationSource(
            citation_id=i + 1,   # 1-indexed，对应 [来源 1]
            excerpt=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
```
- `i + 1`：引用编号从 1 开始（符合人类习惯）
- `excerpt` 截取前 200 字符展示给用户（完整文本太长，摘录即可）

```python
def _format_context(self, chunks):
    for i, chunk in enumerate(chunks):
        part = (
            f"[来源 {citation_num}] {source_line}\n"
            f"{chunk.text}\n"
        )
```
- 给每个块加上编号标注，LLM 在回答时能引用"[来源 1]"等标记
- System prompt 要求 LLM 必须使用这些引用标注

---

## 14. app/generation/generator.py — LLM生成

```python
RAG_SYSTEM_PROMPT = """你是一位专业的医学信息助手...
## 回答规则
1. **严格基于参考资料**：只使用提供的参考资料回答问题
2. **引用来源**：每个关键信息点后面必须用 [来源 X] 标注
...
{context}
"""
```
- System prompt 定义 LLM 的角色和行为约束
- `{context}` 是 Python f-string 占位符，在运行时替换为检索内容
- 强制引用标注：减少幻觉，增加可溯源性

```python
    async def stream(self, question, context, citations, chunks):
        system_content = RAG_SYSTEM_PROMPT.format(context=context)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question),
        ]

        async for chunk in self._llm.astream(messages):
            token_text = chunk.content
            if token_text:
                full_answer += token_text
                yield StreamEvent(event="token", data={"token": token_text})
```
- `self._llm.astream()`：LangChain 的异步流式接口，逐 token 生成
- `chunk.content`：当前 token 的文本内容（可能是 1 个字或多个字）
- `yield`：Python 生成器语法，每生成一个 token 就立即发送给客户端

```python
        yield StreamEvent(event="citations", data={
            "citations": [c.model_dump() for c in citations],
            "self_rag_passed": self_rag_passed,
        })
```
- 答案生成完成后，一次性发送所有引用信息（不需要流式）
- `model_dump()`：Pydantic v2 的序列化方法，将模型转为 dict（可 JSON 化）

```python
    async def _self_rag_verify(self, question, context_summary, answer):
        prompt = SELF_RAG_VERIFICATION_PROMPT.format(...)
        response = await self._verifier_llm.ainvoke(messages)
        verdict = self._parser.invoke(response).strip()
        passed = verdict.startswith("通过")
```
- **Self-RAG**：用另一个 LLM 调用来验证答案是否忠实于参考资料
- 验证 LLM 温度设为 0（确定性输出），只回答"通过"/"不通过"

---

## 15. app/tenants/manager.py — 多租户管理

```python
class TenantManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
```
- **双重检查锁定单例**（Double-Checked Locking）：
  - 第一次 `if`：大多数情况（已初始化）不加锁，性能高
  - `with self._lock`：加锁检查，防止多线程同时初始化
  - 第二次 `if`：加锁后再检查，防止"两个线程都通过第一个 if"的竞态
- `threading.Lock()`：不可重入锁（同一线程不能重复获取）

```python
    def get_or_create(self, tenant_id, description="", **config_kwargs):
        self._validate_tenant_id(tenant_id)
        with self._registry_lock:  # RLock（可重入），因为内部可能再次调用 get
            if tenant_id not in self._tenants:
                config = TenantConfig(...)
                self._tenants[tenant_id] = config
```
- `RLock`（Reentrant Lock）：同一线程可以多次获取，防止自我死锁

```python
    @staticmethod
    def _validate_tenant_id(tenant_id):
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', tenant_id):
            raise ValueError(...)
```
- 安全验证：防止租户 ID 包含路径分隔符（`../`）或特殊字符，避免路径注入攻击
- `^...$`：完全匹配（从行首到行尾），不允许额外字符

---

## 16. app/pipeline.py — 主流水线

```python
class ComponentContainer:
    async def initialize(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:  # 双重检查（异步版）
                return
```
- `asyncio.Lock`（异步锁）：用于协程间的互斥，不会阻塞线程
- 双重检查：防止多个并发请求同时触发初始化

```python
            self._embedding_model = await asyncio.to_thread(
                SentenceTransformer,
                settings.embedding_model_path,
                device="cuda",
            )
```
- `asyncio.to_thread`：将同步的模型加载放到线程池，不阻塞 asyncio 事件循环
- 否则 `SentenceTransformer()` 加载约需 10s，会阻塞所有其他 async 任务

```python
class RAGPipeline:
    async def query_stream(self, request):
        # Step 1: 查询扩展
        yield StreamEvent(event="metadata", data={"status": "expanding_query"})
        query_variants_dict = await self.container.query_processor.expand(...)

        # Step 2: 混合检索
        yield StreamEvent(event="metadata", data={"status": "retrieving"})
        raw_docs = await self.container.retriever.retrieve(...)

        # Step 3: Reranker
        yield StreamEvent(event="metadata", data={"status": "reranking"})
        reranked_chunks = await asyncio.to_thread(
            self.container.reranker.rerank, ...
        )
```
- 每个步骤前先 `yield` 一个状态事件，让前端显示进度（"正在检索..."）
- `reranker.rerank` 是同步方法（CrossEncoder 推理），用 `to_thread` 转异步

---

## 17. app/main.py — FastAPI入口

```python
app = FastAPI(
    title="Medical RAG API",
    docs_url="/docs",    # Swagger UI 自动生成接口文档
    redoc_url="/redoc",  # ReDoc 格式文档
)
```
- FastAPI 自动根据 Pydantic 模型生成 OpenAPI 规范和 Swagger UI

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
app.add_middleware(GZipMiddleware, minimum_size=1000)
```
- CORS 中间件：允许浏览器跨域请求（生产环境应限制为具体域名）
- Gzip 中间件：自动压缩 > 1000 字节的响应（JSON 压缩率约 70%）

```python
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(get_container())  # 后台预加载，不阻塞启动
```
- `create_task`：在事件循环中创建后台任务，与 HTTP 请求处理并发执行
- 这样 API 启动后立即可响应 `/health`，模型在后台继续加载

```python
async def event_stream_generator(request, container):
    pipeline = RAGPipeline(container)
    async for event in pipeline.query_stream(request):
        payload = json.dumps(
            {"event": event.event, "data": event.data},
            ensure_ascii=False,   # 中文字符不转义为 \uXXXX
        )
        yield f"data: {payload}\n\n".encode("utf-8")
```
- `ensure_ascii=False`：JSON 中的中文保持 UTF-8，不转义为 `\u4e2d\u6587`（节省 6x 空间）
- `f"data: {payload}\n\n"`：SSE 标准格式，每个消息以双换行结束
- `.encode("utf-8")`：`StreamingResponse` 需要 bytes，不是 str

```python
@app.post("/query", tags=["问答"])
async def query(request: QueryRequest, container = Depends(get_container)):
    if request.stream:
        return StreamingResponse(
            event_stream_generator(request, container),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",     # 禁缓存（SSE 必须）
                "X-Accel-Buffering": "no",        # 禁 Nginx 缓冲（关键！）
            },
        )
```
- `Depends(get_container)`：FastAPI 依赖注入，自动获取已初始化的组件容器
- `X-Accel-Buffering: no`：告诉 Nginx 不要缓冲响应，立即转发每个 chunk

---

## 18. scripts/ingest.py — 摄入脚本

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```
- 将项目根目录加入 Python 模块搜索路径
- `Path(__file__).parent.parent`：`scripts/ingest.py` → `scripts/` → 项目根目录
- 使 `from app.xxx import yyy` 在脚本中正常工作

```python
embedding_model = SentenceTransformer(
    settings.embedding_model_path,
    device="cuda",
)
```
- 脚本中直接同步加载（不需要异步），因为 ingest 脚本是单线程运行的

```python
entity_extractor = MedicalEntityExtractor(use_spacy=False)  # 脚本中关闭 spaCy
```
- 摄入时关闭 spaCy：加快速度（不需要识别英文实体，节省约 1-2s/1000 块）

---

## 19. 完整数据流讲解

### 阶段一：文档摄入（离线，一次性）

```
《默克诊疗手册.pdf》
    │
    ▼ [PDFProcessor]
    ├── PdfReader 逐页提取文本
    ├── 正则清洗：删除页码行、页眉页脚、软连字符
    └── 识别章节结构（跨页追踪章名/节标题）

    输出：List[{page_number, text, chapter, section_title}]（约 1200 页）
    │
    ▼ [MedicalRuleEngine]（责任链，每个 chunk 依次经过）
    ├── QualityFilterRule：标记低质量块（过短/乱码）
    ├── UnitNormalizationRule：mg/kg → mg·kg⁻¹
    ├── DrugNameNormalizationRule：百忧解 → 氟西汀
    ├── MedicalAbbreviationRule：HTN → 高血压(HTN)
    ├── TableDetectionRule：标记表格块
    └── WarningDetectionRule：标记警告块，设置 importance_boost=1.5

    输出：清洗后的文本 + 丰富的元数据字典
    │
    ▼ [SemanticChunker]
    ├── 按中文标点切分为句子
    ├── BGE-M3 批量 encode 所有句子（GPU，约 300 句/秒）
    ├── 相邻句子余弦相似度 < 0.45 → 语义断点 → 切块
    └── 合并过短块，拆分过长块

    输出：语义完整的文本块
    │
    ▼ [ParentChildChunker]
    ├── 父块（~1000 字符）：语义完整，提供上下文给 LLM
    └── 子块（~300 字符）：精确小块，用于向量检索命中

    输出：List[DocumentChunk]（父块约 2000 个，子块约 8000 个）
    │
    ▼ [MedicalEntityExtractor]（对每个块）
    ├── 词典正则匹配：疾病/药物/症状/体征/实验室指标/操作
    └── 结果写入 chunk.entities 字段
    │
    ▼ [ChromaIndexer]
    ├── 父块 → {prefix}_{tenant}_parents 集合（父块全文 + 向量）
    ├── 子块（过滤低质量）→ {prefix}_{tenant}_children 集合（子块向量）
    └── jieba 分词 → BM25Okapi 构建 → pickle 持久化到磁盘

    存储：Chroma SQLite + HNSW 向量索引 + BM25 pickle 文件
```

### 阶段二：用户查询（实时，流式输出）

```
用户：POST /query {"question": "高血压用什么药？", "tenant_id": "default"}
    │
    ▼ [FastAPI 路由]
    ├── Pydantic 参数验证（question 长度/格式）
    ├── Depends(get_container) → 获取已初始化的组件容器
    └── StreamingResponse(event_stream_generator(...))

    前端收到 SSE 流，开始接收数据...
    │
    ▼ [RAGPipeline.query_stream]
    │
    ── 状态事件 → 前端："正在扩展查询..." ──────────────────────────────
    │
    ▼ [QueryProcessor.expand]（并发 LLM 调用）
    ├── 协程1：原始查询（直接使用）
    │   └── "高血压用什么药？"
    ├── 协程2：查询重写（LLM → DeepSeek API）
    │   └── "高血压的一线降压药物选择与用药方案"
    ├── 协程3：HyDE（LLM 生成假设答案）
    │   └── "高血压治疗的一线药物包括利尿剂、ACE抑制剂、ARB、
    │       钙通道阻滞剂和β受体阻滞剂。根据JNC8指南，大多数
    │       非黑人患者首选噻嗪类利尿剂或ACEI或ARB..."
    └── 协程4：Multi-Query 变体（LLM 生成）
        ├── "降压药物分类及适应症"
        ├── "抗高血压一线用药推荐"
        └── "高血压药物治疗指南"

    asyncio.gather 等待最慢的协程完成（约 1-2s）

    输出：{original: q0, rewritten: q1, hyde: q2, multi_q1: q3, ...}
    │
    ── 状态事件 → 前端："正在检索相关文档..." ───────────────────────────
    │
    ▼ [HybridRetriever.retrieve]（并发对6个查询变体各做双路检索）
    │
    对每个查询变体（并发执行）：
        ┌─────────────────────────────────────────────┐
        │         _retrieve_single(query_i)           │
        │  ┌──────────────────┐ ┌───────────────────┐ │
        │  │   BM25 检索      │ │   向量检索         │ │
        │  │ jieba 分词       │ │ BGE-M3 encode     │ │
        │  │ BM25Okapi 打分   │ │ Chroma HNSW 搜索  │ │
        │  │ Top-20 结果      │ │ Top-20 结果        │ │
        │  └──────────────────┘ └───────────────────┘ │
        │         ↓      Layer1 RRF         ↓         │
        │    {"bm25": [20 docs], "vector": [20 docs]} │
        └─────────────────────────────────────────────┘

    6 个变体并发，每个变体内 BM25+向量 并发，极大降低总延迟
    │
    ▼ [DualRRFFusion.fuse]
    ├── Layer1 RRF（per-query）：
    │   对每个变体的 [bm25_docs, vector_docs] 执行 RRF
    │   → 6 个混合排名列表
    │
    ├── Layer2 RRF（cross-query）：
    │   对 6 个混合列表再次执行 RRF
    │   → 1 个最终候选列表（约 80-120 个文档，含重复）
    │
    └── deduplicate：按 chunk_id 去重
        → 约 50-80 个唯一候选文档
    │
    ▼ [_expand_to_parents]（Parent-Child 上下文扩展）
    对每个子块：查 Chroma parents 集合 → 用父块全文替换
    → 每个候选文档包含完整语义段落（而非截断的小块）
    │
    ── 状态事件 → 前端："正在精排结果..." ───────────────────────────────
    │
    ▼ [BGEReranker.rerank]（CrossEncoder 精确打分）
    ├── 构建 pairs：[(query, doc1), (query, doc2), ...（约 60 对）]
    ├── CrossEncoder.predict：拼接 query+doc 通过 Transformer
    │   → 每对输出一个相关性 logit 值
    ├── 按分数降序排列
    └── 取 Top-8（rerank_top_n）

    输出：List[RetrievedChunk]（8 个高质量块，含精确分数）
    │
    ▼ [ContextProcessor.process]
    ├── compress：过滤 rerank_score < -5.0 的块
    ├── reorder：交替排列 → [最相关, 次次相关, ..., 次相关, 末尾相关]
    │             （Lost in the Middle 缓解）
    ├── truncate：总字符不超过 12000
    ├── build_citations：为每块分配 [来源 1]~[来源 8] 编号
    └── format_context：
        "[来源 1] 章节：心血管疾病 | 小节：高血压治疗 | 页码：第245页
         ...（块文本）...
         ---
         [来源 2] ..."

    输出：(context_str, List[CitationSource])
    │
    ▼ [MedicalEntityExtractor.extract_from_query]
    ├── 口语化映射："血压高" → "高血压"
    └── 词典匹配：{"疾病": ["高血压"], "药物": [...]}
    │
    ── 状态事件 → 前端："正在生成回答..." ───────────────────────────────
    │
    ▼ [RAGGenerator.stream]
    ├── 构建 Prompt：
    │   System: "你是医学助手...{context_str}..."
    │   Human: "高血压用什么药？"
    │
    ├── DeepSeek.astream() 流式生成：
    │   ▶ token: "高血压" → SSE: data: {"event":"token","data":{"token":"高血压"}}
    │   ▶ token: "的" → SSE: data: {"event":"token","data":{"token":"的"}}
    │   ▶ token: "一线" → ...
    │   ▶ token: "[来源1]" → ...
    │   ... 每个 token 立即推送，约 1200 个 token/min
    │
    ├── Self-RAG 验证（答案生成完成后）：
    │   "所给资料是否支持此答案？" → "通过"
    │
    ├── SSE: data: {"event":"citations","data":{citations:[...], self_rag_passed:true}}
    ├── SSE: data: {"event":"metadata","data":{"elapsed_seconds":3.2,...}}
    └── SSE: data: {"event":"done","data":{"message":"生成完成"}}

    前端接收 "done" 事件，关闭 SSE 连接
    │
    ▼ 用户看到的最终答案（示例）：

    "高血压的一线治疗药物主要包括以下几类：

    1. **噻嗪类利尿剂**（如氢氯噻嗪）：通过促进肾脏排钠利尿降低血压 [来源 1]
    2. **ACE 抑制剂**（如依那普利、卡托普利）：抑制血管紧张素转换酶 [来源 2]
    3. **ARB 类**（如坎地沙坦）：适用于对 ACEI 不耐受的患者 [来源 3]
    4. **钙通道阻滞剂**（如氨氯地平）：扩张血管，常用于老年患者 [来源 1]

     注意：降压药物的选择需根据患者合并症（如糖尿病、慢性肾病）
    进行个体化调整 [来源 4]。请在医生指导下用药。"

    附：引用来源
    [来源 1] 心血管疾病 > 高血压 > 治疗 | 第245-247页
    [来源 2] 心血管疾病 > 高血压 > 药物选择 | 第248页
    ...
```


*文档生成时间：2026-03-25 | 系统版本：1.0.0*
