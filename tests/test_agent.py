"""单元测试：验证知识库检索与 Agent 逻辑（不依赖真实 LLM API）。"""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent))

from agent import CustomerServiceAgent, KnowledgeBase, LLMClient  # noqa: E402


class FakeLLM(LLMClient):
    """测试用 LLM：直接回显上下文，不调用真实 API。"""

    def chat(self, messages, temperature=0.3):
        user_msg = messages[-1]["content"]
        return f"[mock] 基于上下文回答: {user_msg[:50]}"


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(
            [
                {"q": "营业时间", "a": "周一至周五 9:00-18:00"},
                {"q": "如何退款", "a": "订单页申请退款"},
                {"text": "会员可享积分累计"},
            ]
        )

    def test_search_hit(self):
        hits = self.kb.search("你们几点营业")
        self.assertTrue(len(hits) >= 1)
        top = hits[0][0]
        self.assertIn("营业时间", top.get("q", ""))

    def test_search_miss(self):
        hits = self.kb.search("完全无关的问题xyz")
        self.assertEqual(hits, [])

    def test_add_entry(self):
        self.kb.add(q="配送时间", a="1-2天")
        self.assertEqual(len(self.kb.entries), 4)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(
            [{"q": "营业时间", "a": "周一至周五 9:00-18:00"}]
        )
        self.agent = CustomerServiceAgent(self.kb, FakeLLM())

    def test_answer_hit(self):
        result = self.agent.answer_with_sources("几点营业")
        self.assertIn("mock", result["answer"])
        self.assertTrue(len(result["sources"]) >= 1)

    def test_answer_miss(self):
        result = self.agent.answer_with_sources("今天天气怎么样")
        self.assertIn("没有查到", result["answer"])
        self.assertEqual(result["sources"], [])


class TestLoadJson(unittest.TestCase):
    def test_load_faq(self):
        kb_path = (
            Path(__file__).parent.parent / "knowledge_base" / "faq.json"
        )
        kb = KnowledgeBase().load_json(str(kb_path))
        self.assertGreater(len(kb.entries), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
