"""命令行入口：交互式问答演示。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import CustomerServiceAgent, KnowledgeBase, LLMClient  # noqa: E402


def main() -> None:
    kb_path = Path(__file__).parent / "knowledge_base" / "faq.json"
    kb = KnowledgeBase().load_json(str(kb_path))
    agent = CustomerServiceAgent(kb, LLMClient())

    print("=" * 50)
    print("智能客服问答 Agent 已启动")
    print("知识库已加载，共", len(kb.entries), "条知识")
    print("输入问题开始对话，输入 exit / quit 退出")
    print("=" * 50)

    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break
        try:
            result = agent.answer_with_sources(query)
            print(f"\n客服：{result['answer']}")
        except RuntimeError as e:
            print(f"\n[错误] {e}")
            break


if __name__ == "__main__":
    main()
