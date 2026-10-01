"""智能客服问答 Agent 核心逻辑。

设计目标：零第三方重依赖（除 LLM API），可独立运行、可测试。
流程：用户问题 -> 知识库检索（TF-IDF 相似度）-> 命中则基于上下文生成回答 -> 未命中则走兜底。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """基于 TF-IDF 的轻量知识库，无需向量数据库即可检索。

    每条知识是一条 (question, answer) 或纯文本条目。
    """

    def __init__(self, entries: List[dict] | None = None):
        self.entries: List[dict] = entries or []

    def load_json(self, path: str) -> "KnowledgeBase":
        """从 JSON 文件加载知识条目。

        支持两种格式：
        [{"q": "问题", "a": "答案"}, ...]  或  [{"text": "一段知识"}, ...]
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = data
        return self

    def add(self, q: str = "", a: str = "", text: str = "") -> None:
        if text:
            self.entries.append({"text": text})
        else:
            self.entries.append({"q": q, "a": a})

    def _tokenize(self, text: str) -> List[str]:
        """极简分词：按非中文字符切分，兼顾英文按空格切分。"""
        import re
        # 中文连续字符 + 英文单词 + 数字
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text.lower())
        return tokens

    @staticmethod
    def _char_bigrams(text: str) -> set:
        """中文按字符 bigram 切分，英文按单词。对短查询鲁棒性更好。"""
        import re
        text = text.lower()
        tokens: List[str] = []
        for chunk in re.findall(r"[\u4e00-\u9fff]+|[a-z]+|\d+", text):
            if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
                # 中文字符 bigram
                tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
                tokens.append(chunk)  # 整串也算一个特征，捕获单字词
            else:
                tokens.append(chunk)
        return set(tokens)

    def search(self, query: str, top_k: int = 3, threshold: float = 0.1) -> List[Tuple[dict, float]]:
        """返回与 query 最相关的 top_k 条知识及相似度分数。

        相似度 = 字符 bigram 集合的 Jaccard 相似度，短查询召回更稳。
        """
        if not self.entries:
            return []
        scored: List[Tuple[dict, float]] = []
        q_grams = self._char_bigrams(query)
        if not q_grams:
            return []
        for entry in self.entries:
            text = entry.get("q", "") + " " + entry.get("a", "") + " " + entry.get("text", "")
            e_grams = self._char_bigrams(text)
            if not e_grams:
                score = 0.0
            else:
                overlap = len(q_grams & e_grams)
                jaccard = overlap / len(q_grams | e_grams) if (q_grams | e_grams) else 0.0
                # 回退：仅当 query 中存在「长度>=2」的 gram 完整命中知识才算字面命中，
                # 排除单字（如"天"）造成的误召回。
                containment = any(len(g) >= 2 and g in text for g in q_grams)
                containment_score = 0.15 if containment else 0.0
                score = max(jaccard, containment_score)
            scored.append((entry, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 过滤低于阈值的
        filtered = [(e, s) for e, s in scored[:top_k] if s >= threshold]
        return filtered


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

class LLMClient:
    """可配置的大模型客户端，默认对接智谱 GLM（OpenAI 兼容协议）。

    通过环境变量配置：
    - LLM_API_KEY: API Key
    - LLM_API_BASE: 接口地址（默认智谱 https://open.bigmodel.cn/api/paas/v4）
    - LLM_MODEL: 模型名（默认 glm-4-flash）
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.api_base = (
            api_base
            or os.environ.get("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "glm-4-flash")

    def chat(self, messages: List[dict], temperature: float = 0.3) -> str:
        """调用聊天补全接口，返回模型回复文本。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 LLM_API_KEY。请设置环境变量 LLM_API_KEY 后再运行。"
            )
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Agent 主逻辑
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一名专业的智能客服助手。请基于给定的知识库内容回答用户问题。"
    "回答要准确、简洁、礼貌。如果知识库中没有相关信息，请明确说明"
    "'抱歉，我暂时没有查到相关信息，建议您联系人工客服。'，"
    "不要编造知识库中不存在的内容。"
)


class CustomerServiceAgent:
    """智能客服 Agent：检索 + 生成。"""

    def __init__(self, kb: KnowledgeBase, llm: LLMClient | None = None):
        self.kb = kb
        self.llm = llm or LLMClient()

    def _build_context(self, query: str) -> str:
        hits = self.kb.search(query, top_k=3)
        if not hits:
            return ""
        parts = []
        for entry, score in hits:
            if "text" in entry:
                parts.append(entry["text"])
            else:
                parts.append(f"Q: {entry.get('q', '')}\nA: {entry.get('a', '')}")
        return "\n\n".join(parts)

    def answer(self, query: str) -> str:
        """返回对用户问题的回答。"""
        context = self._build_context(query)
        if not context:
            return "抱歉，我暂时没有查到相关信息，建议您联系人工客服。"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"知识库内容：\n{context}\n\n用户问题：{query}"},
        ]
        return self.llm.chat(messages)

    def answer_with_sources(self, query: str) -> dict:
        """返回带检索来源的回答（便于调试和展示）。"""
        hits = self.kb.search(query, top_k=3)
        context = self._build_context(query)
        answer = self.answer(query)
        return {"query": query, "sources": hits, "answer": answer}
