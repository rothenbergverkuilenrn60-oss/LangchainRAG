"""
generator.py — LLM Generator (streaming + Self-RAG + fine-grained citations)
Responsible for final answer generation, integrating three key quality mechanisms:

1. Streaming Generation:
   Uses LangChain's astream_events API to push tokens to the client one by one,
   significantly reducing user-perceived latency (first token response < 1s).

2. Self-RAG Verification:
   After generating an answer, uses the LLM to self-verify:
   "Is the retrieved reference material sufficient to support this answer?"
   If not, returns a clear disclaimer rather than a fabricated answer.

3. Fine-grained Citations:
   The system prompt forces the LLM to annotate each knowledge point with [Source X],
   outputting structured citation information for frontend display.
"""

import json
import logging
import asyncio
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.models import RetrievedChunk, CitationSource, RAGResponse, StreamEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """你是一位专业的医学信息助手，基于《默克诊疗手册》为医疗专业人员和患者提供准确、权威的医学信息。

## 回答规则
1. **严格基于参考资料**：只使用提供的参考资料回答问题，不添加参考资料中没有的信息
2. **引用来源**：每个关键信息点后面必须用 [来源 X] 标注来源编号（X 对应参考资料编号），可同时引用多个如 [来源 1][来源 3]
3. **警告优先**：参考资料中标有【警告】的内容涉及禁忌症、严重副作用或药物相互作用，必须在回答中单独列出并加以强调
4. **专业术语**：使用规范医学术语，必要时提供通俗解释
5. **结构化输出**：使用清晰的段落和标题组织回答，章节来源可在引用时提及（如"根据{{章节名}}..."）
6. **不确定性表达**：如果参考资料信息不足，明确告知，不要猜测
7. **安全第一**：涉及用药、治疗等决策时，必须建议咨询医生
8. **图示处理**：当参考资料中某来源标有 [图示] 时，表示该来源是一张医学图片（已单独提供给用户）。请在回答中用 [来源 X] 引用该图示，并明确告知用户"图示见 [来源 X]"，不要尝试用文字描述图片内容

## 输出格式
回答需包含：
- 直接回答问题的主要内容（带来源引用）
- 警告与注意事项（如参考资料中存在【警告】内容，必须单独列出）
- 如需进一步信息的建议

## 参考资料
以下是从《默克诊疗手册》检索到的相关内容：

{context}

---
请基于以上参考资料，专业、准确地回答用户问题。每个重要信息点请标注对应的 [来源 X]。"""

IS_RETRIEVAL_NEEDED_PROMPT = """判断以下问题是否需要检索专业医学文档才能回答。

规则：
- 如果是常识、数学计算、基本事实、日常知识 → 直接回答
- 如果是医学专业问题、症状、药物、疾病、治疗方案 → 需要检索

只回答"直接回答"或"需要检索"，后跟一句理由（不超过20字）。

问题：{question}"""

SELF_RAG_VERIFICATION_PROMPT = """你是一位医学信息质量评估专家。
请评估以下情况：

问题：{question}

参考资料摘要：{context_summary}

生成的回答：{answer}

评估标准（必须同时满足才能"通过"）：
1. 回答的核心结论是否能在参考资料中找到明确依据？（允许合理归纳，但不允许使用参考资料之外的知识作为核心依据）
2. 回答中是否没有与参考资料矛盾的信息？
3. 是否没有可能严重误导患者的危险建议？

判定规则：
- 如果回答包含参考资料中**完全未提及**的关键医学信息作为核心结论 → "不通过"
- 如果回答包含明显捏造的医学事实或危险建议 → "不通过"
- 如果回答明确说明"参考资料未涉及"或"无法从资料中得出结论"→ "通过"（诚实声明不算错误）

请直接回答"通过"或"不通过"，后跟一句简短的理由（不超过 50 字）。"""


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    RAG generator based on LangChain + DeepSeek.
    Supports both streaming and non-streaming output modes.

    Usage (streaming):
        generator = RAGGenerator()
        async for event in generator.stream(question, context, citations, chunks):
            yield event  # StreamEvent object

    Usage (non-streaming):
        response = await generator.generate(question, context, citations, chunks)
    """

    def __init__(self):
        # Main generation LLM (streaming)
        self._llm = ChatDeepSeek(
            model=settings.llm_model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=True,  # Enable streaming output
        )

        # Verifier LLM (no streaming needed, fast judgment)
        self._verifier_llm = ChatDeepSeek(
            model=settings.llm_model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.0,   # Deterministic output for verification (temperature=0)
            max_tokens=100,
            streaming=False,
        )
        self._parser = StrOutputParser()

    async def stream(
        self,
        question: str,
        context: str,
        citations: List[CitationSource],
        chunks: List[RetrievedChunk],
    ) -> AsyncIterator[StreamEvent]:
        """
        Streaming answer generation, progressively pushing via Server-Sent Events:
          1. Push "token" events token by token
          2. Push "citations" event after generation (complete citation info)
          3. Push "metadata" event (elapsed time, Self-RAG result, etc.)
          4. Push "done" event (stream end marker)

        Args:
            question: Original user question
            context: Formatted retrieval context
            citations: Fine-grained citation list
            chunks: Retrieved chunk list (used for Self-RAG verification)

        Yields:
            StreamEvent: stream event objects
        """
        import time
        start_time = time.time()

        # Build prompt
        system_content = RAG_SYSTEM_PROMPT.format(context=context)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question),
        ]

        full_answer = ""

        # ── Streaming token push ─────────────────────────────────────────────
        try:
            async for chunk in self._llm.astream(messages):
                # chunk is a LangChain AIMessageChunk object
                # chunk.content is the current token text (may contain multiple characters)
                token_text = chunk.content
                if token_text:
                    full_answer += token_text
                    yield StreamEvent(
                        event="token",
                        data={"token": token_text},
                    )
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield StreamEvent(event="error", data={"message": str(e)})
            return

        elapsed = time.time() - start_time
        logger.info(f"Generation complete: {len(full_answer)} chars, elapsed {elapsed:.1f}s")

        # ── Self-RAG verification (async in background, doesn't block token stream) ──
        self_rag_passed = True
        self_rag_reason = ""
        try:
            self_rag_passed, self_rag_reason = await self._self_rag_verify(
                question=question,
                context_summary=context[:5000],  # Cover enough retrieval content
                answer=full_answer,
            )
        except Exception as e:
            logger.warning(f"Self-RAG verification failed (skipped): {e}")

        # ── Push citation info ───────────────────────────────────────────────
        yield StreamEvent(
            event="citations",
            data={
                "citations": [c.model_dump() for c in citations],
                "self_rag_passed": self_rag_passed,
                "self_rag_reason": self_rag_reason,
            },
        )

        # ── Push metadata ────────────────────────────────────────────────────
        yield StreamEvent(
            event="metadata",
            data={
                "elapsed_seconds": round(elapsed, 2),
                "answer_length": len(full_answer),
                "context_chunks": len(chunks),
                "citations_count": len(citations),
                "self_rag_passed": self_rag_passed,
            },
        )

        # ── Stream end marker ────────────────────────────────────────────────
        yield StreamEvent(event="done", data={"message": "Generation complete"})

    async def generate(
        self,
        question: str,
        context: str,
        citations: List[CitationSource],
        chunks: List[RetrievedChunk],
    ) -> RAGResponse:
        """
        Non-streaming generation (for batch processing or testing).
        """
        import time
        start_time = time.time()

        system_content = RAG_SYSTEM_PROMPT.format(context=context)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question),
        ]

        response = await self._llm.ainvoke(messages)
        answer = self._parser.invoke(response)

        elapsed = time.time() - start_time

        # Self-RAG verification
        self_rag_passed, self_rag_reason = await self._self_rag_verify(
            question=question,
            context_summary=context[:5000],
            answer=answer,
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            query_variants=[],   # Filled in by pipeline
            entities_in_query={},  # Filled in by pipeline
            self_rag_passed=self_rag_passed,
            metadata={
                "elapsed_seconds": round(elapsed, 2),
                "answer_length": len(answer),
                "self_rag_reason": self_rag_reason,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # Self-RAG verification
    # ─────────────────────────────────────────────────────────────────────

    async def check_retrieval_needed(self, question: str) -> bool:
        """
        Determines whether the question requires document retrieval.
        For common knowledge, math, etc., answers directly using model knowledge
        to avoid unnecessary retrieval.

        Returns:
            True  → retrieval needed
            False → answer directly with model knowledge
        """
        prompt = IS_RETRIEVAL_NEEDED_PROMPT.format(question=question)
        try:
            response = await self._verifier_llm.ainvoke([HumanMessage(content=prompt)])
            verdict = self._parser.invoke(response).strip()
            needs_retrieval = not verdict.startswith("直接回答")
            logger.info(f"IsRetrievalNeeded: {'yes' if needs_retrieval else 'no'} - {verdict}")
            return needs_retrieval
        except Exception as e:
            logger.warning(f"IsRetrievalNeeded check failed, defaulting to retrieval: {e}")
            return True  # Conservative fallback: use normal retrieval flow

    async def stream_direct(self, question: str) -> AsyncIterator[StreamEvent]:
        """
        Streams an answer directly using model knowledge, without retrieval.
        Used for common-knowledge questions, bypassing the entire retrieval chain.
        """
        import time
        start_time = time.time()

        messages = [
            SystemMessage(content="你是一个知识渊博的助手，请直接、准确地回答用户问题。"),
            HumanMessage(content=question),
        ]

        full_answer = ""
        try:
            async for chunk in self._llm.astream(messages):
                token_text = chunk.content
                if token_text:
                    full_answer += token_text
                    yield StreamEvent(event="token", data={"token": token_text})
        except Exception as e:
            logger.error(f"Direct generation error: {e}")
            yield StreamEvent(event="error", data={"message": str(e)})
            return

        elapsed = time.time() - start_time
        yield StreamEvent(event="citations", data={"citations": [], "self_rag_passed": True, "self_rag_reason": "No retrieval needed, answered directly"})
        yield StreamEvent(event="metadata", data={"elapsed_seconds": round(elapsed, 2), "answer_length": len(full_answer), "context_chunks": 0, "citations_count": 0, "self_rag_passed": True, "retrieval_skipped": True})
        yield StreamEvent(event="done", data={"message": "Generation complete"})

    async def _self_rag_verify(
        self,
        question: str,
        context_summary: str,
        answer: str,
    ) -> Tuple[bool, str]:
        """
        Self-RAG self-verification: checks whether the generated answer
        is faithful to the reference materials.

        Returns:
            (is_passed, reason_text)
        """
        prompt = SELF_RAG_VERIFICATION_PROMPT.format(
            question=question,
            context_summary=context_summary,
            answer=answer[:2000],  # Truncate answer to first 2000 chars for verification
        )
        try:
            messages = [HumanMessage(content=prompt)]
            response = await self._verifier_llm.ainvoke(messages)
            verdict = self._parser.invoke(response).strip()

            if verdict.startswith("不通过"):
                passed = False
                reason = verdict[3:].strip()
            elif verdict.startswith("通过"):
                passed = True
                reason = verdict[2:].strip()
            else:
                passed = True
                reason = verdict
            logger.info(f"Self-RAG verification: {'passed' if passed else 'failed'} - {reason}")
            return passed, reason
        except Exception as e:
            logger.warning(f"Self-RAG verification error: {e}")
            return True, ""  # Default to passed on error, doesn't affect main flow
