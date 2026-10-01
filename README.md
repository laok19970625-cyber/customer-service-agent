# 智能客服问答 Agent

一个基于「检索 + 生成」（RAG）的轻量智能客服系统：从知识库中检索相关内容，再交给大模型生成准确、克制的回答。零第三方重依赖，双击即可运行。

## 特性

- **零重依赖**：仅用 Python 标准库实现知识库检索（TF-IDF / Jaccard 相似度），无需安装向量数据库。
- **可配置大模型**：默认对接智谱 GLM（OpenAI 兼容协议），可一键切换到 OpenAI / 通义 / DeepSeek 等任何兼容接口。
- **拒绝幻觉**：检索不到相关内容时明确提示联系人工客服，不编造答案。
- **可测试**：内置单元测试，逻辑与 API 解耦。

## 快速开始

### 1. 配置 API Key

```bash
export LLM_API_KEY="你的智谱/OpenAI Key"
# 可选：切换接口
export LLM_API_BASE="https://open.bigmodel.cn/api/paas/v4"
export LLM_MODEL="glm-4-flash"
```

### 2. 运行

```bash
python main.py
```

进入交互式问答，输入问题即可获得回答。

### 3. 运行测试

```bash
python -m unittest discover -s tests -v
```

## 目录结构

```
customer-service-agent/
├── agent.py              # 核心逻辑：知识库 + LLM + Agent
├── main.py               # 命令行交互入口
├── knowledge_base/
│   └── faq.json          # 知识库示例（可替换为你自己的数据）
└── tests/
    └── test_agent.py     # 单元测试
```

## 如何接入你自己的知识库

编辑 `knowledge_base/faq.json`，支持两种格式：

```json
[
  {"q": "问题", "a": "答案"},
  {"text": "一段自由文本知识"}
]
```

## 工作原理

```
用户提问
   │
   ▼
知识库检索（Jaccard 相似度，取 Top-K）
   │
   ├─ 命中 → 构造上下文 → LLM 生成回答
   │
   └─ 未命中 → 返回「建议联系人工客服」
```

## 应用场景

- 电商 / 企业官网自动客服
- 内部员工 FAQ 机器人
- 售后 / 政策咨询自助应答

## License

MIT
