"""
query_processor.py — Multi-strategy query processor
Expands the original user question into multiple optimized query variants to improve recall:

  1. Original Query: Uses the user input directly
  2. Query Rewrite: LLM optimizes colloquial/vague queries into precise medical search terms
  3. HyDE (Hypothetical Document Embedding):
     LLM first generates a hypothetical "ideal answer", then uses its embedding for retrieval
     — because the answer's semantic space is closer to documents than the raw question
  4. Multi-Query:
     LLM generates 3 query variants from different angles, retrieves in parallel then aggregates

All variants are generated asynchronously in parallel to keep latency acceptable.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.models import QueryMode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """你是一位专业的医学信息检索专家。
将用户的问题改写为更精确的医学检索查询，要求：
1. 使用规范医学术语（如将"发烧"改为"发热"）
2. 明确疾病/症状/药物的具体名称
3. 去除口语化表达
4. 保持问题的核心意图不变
直接输出改写后的查询，不要任何解释。"""

HYDE_SYSTEM = """你是一位权威的临床医学专家，正在为医学手册撰写内容。
请针对以下医学问题，写一段简洁、专业的参考答案（100-200字）。
要求：使用规范医学术语，涵盖关键概念，像教科书内容一样客观。
直接输出答案内容，不要任何前言。"""

MULTI_QUERY_SYSTEM = """你是医学检索专家。
针对以下问题，从不同角度生成{count}个检索查询变体。
要求：
1. 每个变体用不同词汇/角度表达同一意图
2. 包含同义词、相关术语、上位概念
3. 每行输出一个查询，不加编号和多余说明
直接输出{count}行查询，不要其他内容。"""


# ─────────────────────────────────────────────────────────────────────────────
# Query processor
# ─────────────────────────────────────────────────────────────────────────────

class QueryProcessor:
    """
    Multi-strategy query processor: expands user questions into multiple optimized queries.

    Usage:
        processor = QueryProcessor()
        variants = await processor.expand(question, mode=QueryMode.full)
        # variants = {
        #   "original": "...",
        #   "rewritten": "...",
        #   "hyde": "...",
        #   "multi_queries": ["...", "...", "..."],
        #   "all": [...]  # Deduplicated complete list
        # }
    """

    def __init__(self):
        # Initialize LLM (DeepSeek)
        self._llm = ChatDeepSeek(
            model=settings.llm_model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.3,        # Slightly higher temperature for query generation to increase diversity
            max_tokens=512,         # Query variants don't need long output
        )
        self._parser = StrOutputParser()  # Parses LLM output into plain string

    async def expand(
        self,
        question: str,
        mode: QueryMode = QueryMode.full,
    ) -> Dict[str, Any]:
        """
        Asynchronously generates all query variants.

        Args:
            question: Original user question
            mode: Controls which expansion strategies are enabled

        Returns:
            dict containing various variants and the "all" list
        """
        result = {
            "original": question,
            "rewritten": question,
            "hyde": question,
            "multi_queries": [],
            "all": [],
        }

        if mode == QueryMode.basic:
            # Basic mode: no expansion
            result["all"] = [question]
            return result

        # ── Concurrently generate multiple variants ──────────────────────────
        tasks = []
        task_names = []

        # Always perform query rewrite
        tasks.append(self._rewrite_query(question))
        task_names.append("rewritten")

        if mode == QueryMode.full:
            # Full mode: additionally generate HyDE and Multi-Query
            tasks.append(self._generate_hyde(question))
            task_names.append("hyde")

            tasks.append(self._generate_multi_queries(question))
            task_names.append("multi_queries")

        # Execute all LLM calls concurrently (asyncio.gather sends requests simultaneously)
        try:
            # return_exceptions=True: even if a task throws an exception, gather won't abort;
            # instead returns the exception object as the result
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Query expansion concurrent execution error: {e}")
            results = [question] * len(tasks)

        # Process results
        for name, res in zip(task_names, results):
            if isinstance(res, Exception):
                logger.warning(f"Query variant generation failed [{name}]: {res}")
                result[name] = question if name != "multi_queries" else []
            else:
                result[name] = res

        # ── Aggregate and deduplicate ─────────────────────────────────────────
        all_queries = [question]  # Original query always goes first

        if result["rewritten"] and result["rewritten"] != question:
            all_queries.append(result["rewritten"])

        if result.get("hyde") and result["hyde"] != question:
            all_queries.append(result["hyde"])

        for q in result.get("multi_queries", []):
            if q and q not in all_queries:
                all_queries.append(q)

        result["all"] = all_queries
        logger.info(f"Query expansion complete: {len(all_queries)} variants")
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Private methods: strategy implementations
    # ─────────────────────────────────────────────────────────────────────

    async def _rewrite_query(self, question: str) -> str:
        """
        Query rewrite: rewrites colloquial/vague questions into standard medical search terms.
        """
        try:
            messages = [
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(content=f"原始问题：{question}"),
            ]
            # ainvoke is LangChain's async invocation interface
            response = await self._llm.ainvoke(messages)
            rewritten = self._parser.invoke(response).strip()
            logger.debug(f"Query rewrite: [{question}] → [{rewritten}]")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return question

    async def _generate_hyde(self, question: str) -> str:
        """
        HyDE: generates a hypothetical answer document.
        Core idea: the answer's semantic space is closer to relevant documents,
        so retrieval using the answer embedding outperforms retrieval using the question embedding.
        """
        try:
            messages = [
                SystemMessage(content=HYDE_SYSTEM),
                HumanMessage(content=question),
            ]
            response = await self._llm.ainvoke(messages)
            hyde_doc = self._parser.invoke(response).strip()
            logger.debug(f"HyDE generation complete, length: {len(hyde_doc)} chars")
            return hyde_doc
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return question

    async def _generate_multi_queries(self, question: str) -> List[str]:
        """
        Multi-Query: generates multiple query variants from different angles.
        Each variant covers different vocabulary/conceptual angles to maximize recall.
        """
        count = settings.multi_query_count
        try:
            system_prompt = MULTI_QUERY_SYSTEM.format(count=count)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
            response = await self._llm.ainvoke(messages)
            raw_output = self._parser.invoke(response).strip()

            # Split by line, filter empty lines (prompt convention: one query per line)
            queries = [
                line.strip()
                for line in raw_output.split('\n')
                if line.strip() and len(line.strip()) > 3
            ][:count]  # Take at most count queries

            logger.debug(f"Multi-Query generation: {queries}")
            return queries
        except Exception as e:
            logger.warning(f"Multi-Query generation failed: {e}")
            return []
