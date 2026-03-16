# TianChi-Research-Agent

> **🏆 阿里云天池大赛 —「寻找AI全能王」Data+AI 工程师全球大奖赛 · 高校赛道 第二名**

一个面向**多跳知识问答**的自主搜索 Agent。给定一个可能需要多步推理、跨源检索才能回答的复杂问题，Agent 自主规划搜索策略、抓取网页、逐步推理，最终输出精确答案。

核心思路是 **ReAct 单循环架构**：LLM 作为推理与决策核心，在每一轮中思考（Thought）→ 选择动作（Action: 搜索 / 抓取 / 完成）→ 获取观察（Observation），循环往复直到答案收敛。在此基础上叠加了三阶段证伪验证、结构化 Findings 记忆、幻觉检测、换答案门槛控制、超时候选回退等工程化增强，使 Agent 在开放域多跳问题上具备较高的准确率与鲁棒性。

## 目录

- [Features](#features)
- [Architecture](#architecture)
- [Prompt Engineering](#prompt-engineering)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- **ReAct 单循环架构** — Thought → Action → Observation 循环，LLM 自主决策搜索、抓取、完成
- **三阶段证伪验证** — 首次 finish → 强制证伪搜索 → 最终确认，防止未经验证的答案直接输出
- **结构化 Findings 记忆** — `Finding: key = val` 持久化关键事实，不受 Trace 窗口压缩影响
- **幻觉检测与防御** — 自动识别 LLM 自造的 Observation/Result，截断并注入警告
- **换答案门槛控制** — 前 2 次允许切换，第 3 次起 BLOCK，连续 3 次 BLOCK 触发死循环逃逸
- **超时候选回退** — 全程累积 `answer_candidates`，超时后四阶段评分提取最优答案
- **9 条 Prompt 策略** — 约束分解、搜索多样性、证据优先、证伪挑战等，每条策略均由真实失败案例驱动
- **智能 Trace 压缩** — 近 5 步完整保留，旧步骤省略冗长 Observation，关键事实已在 Findings 中

## Architecture

```text
POST {"question": "..."}
 │
 ▼
agent.py ── FastAPI 路由，SSE Ping 保活
 │
 ▼
agent_loop.py ── 桥接层
 │
 ▼
react_agent.py ── ReAct 推理循环（≤50 轮）
 │
 ├─ search(query) → search_provider.py → Serper Google 搜索
 ├─ fetch(url)    → page_fetcher.py    → Jina Reader / aiohttp 抓取
 └─ finish(answer)→ 三阶段验证状态机 → 答案格式清洗 → 返回
```

每轮循环：

1. **守卫检查**：剩余时间 < 15s 或预算耗尽 → 强制退出
2. **组装 Prompt**：注入问题 + Findings + 历史 Trace + 剩余预算
3. **调用 LLM**：`temperature=0`，动态超时 `min(LLM_TIMEOUT, max(30, remaining × 0.4))`
4. **解析输出**：提取 Thought / Action / Action Input / Finding / 幻觉标志
5. **执行 Action**：搜索 / 抓取 / 验证完成
6. **追加 Trace**：记录本轮完整推理链

### 相对比ReAct

| 维度 | 纯 ReAct | 本系统 |
|------|---------|--------|
| **答案验证** | LLM 自行决定何时输出 | 三阶段证伪状态机 |
| **中间记忆** | 仅靠上下文窗口 | Findings 结构化记录，始终注入 |
| **幻觉防御** | 无 | 检测自造内容 + 截断 + 警告注入 |
| **答案稳定性** | 可任意切换 | 门槛控制 + 死循环逃逸 |
| **超时容错** | 超时即丢失 | 四阶段评分回退提取 |
| **Trace 管理** | 全量或固定窗口 | 智能压缩：近步完整 + 旧步精简 |
| **搜索质量** | 自由生成查询 | 重复拦截 + 多样性引导 + 查询压缩 |

## Prompt Engineering

> **核心点：ReAct Agent 的准确率瓶颈在 Prompt，而非代码架构。** 代码层提供循环骨架和安全网，但搜索什么、怎么推理、何时停止——这些决策完全由 Prompt 驱动。

系统 Prompt（`react_prompt.py`）包含 **9 条策略**，每条策略都由真实的失败案例驱动迭代而来。以下按逻辑顺序说明设计理念。

### 策略总览

| # | 策略 | 核心理念 | 解决的问题 |
|---|------|---------|-----------|
| 1 | Decompose first | 约束枚举 + 最强判别约束优先 | 多约束问题遗漏关键条件 |
| 2 | Search effectively | 搜索多样性 + 维度轮换 | 重复搜索浪费预算、搜索死循环 |
| 3 | Evidence over intuition | 证据优先，禁止内部知识否决 | LLM 用"常识"推翻搜索证据 |
| 4 | Cross-validate | 多源交叉验证 | 单源信息可能有误 |
| 5 | Challenge your hypothesis | 证伪搜索 + 换答案门槛 | 首个候选即锁定、答案反复摇摆 |
| 6 | Multi-hop reasoning | 链式推理 + Finding 记录 | 中间结论丢失导致推理断裂 |
| 7 | When to fetch | 百科路由策略 | 盲目抓取浪费预算 |
| 8 | When and how to finish | 全约束复查 + 语言检查 | 遗漏约束导致答案错误 |
| 9 | Answer format | 完整名称 + 原文名 + 多实体 | 答案不完整或格式错误 |

### 策略 1：Decompose first — 约束枚举与优先排序

**问题**：复杂问题通常包含 3-5 个约束条件（时间、地点、属性、关系等）。LLM 倾向于抓住最显眼的约束搜索，忽略隐含的判别性约束。

**设计**：
- 第一轮 Thought 必须列出所有约束并编号（SQ1, SQ2, SQ3...）
- 标记**最强判别约束**（能排除最多候选的那个）优先搜索
- 技术/科学约束 > 常识约束（如"基因拷贝数变异控制季节型"比"副产物用作调味料"更具排除力）
- 歧义约束要列出所有解释（如"功率单位"：瓦特/马力/千瓦；"马力"既是人名也是单位）
- **前 1-2 次搜索只用问题中的约束，不预设候选实体**，避免确认偏误

### 策略 2：Search effectively — 搜索多样性

**问题**：LLM 倾向于反复用相似关键词搜索，换个同义词就认为是"新搜索"，导致预算浪费。

**设计**：
- **精简查询**：2-5 个核心名词/名称，≤40 字符，删除所有填充词（形容词、动词、元词如 history/related）
- **实质性差异**：每次搜索必须包含至少一个从未使用过的核心关键词，仅换序/换同义词不算不同
- **渐进式精炼**：从 2-3 关键词开始，结果太多则加一个词，结果太少则减词，**不要用更长的查询应对差结果**
- **语言切换强制**：同一语言 2 次无结果后，下一次必须换语言
- **维度轮换清单**（卡住时按序尝试）：
  1. 将描述转译为可能的标题/短语
  2. 隔离单一约束，仅用 2 词搜索
  3. 追踪 snippet 中提到的陌生实体名
  4. 反向搜索（从结果找原因）
  5. 搜索列表/类别
  6. 换语言（德语/法语/日语等）
  7. Fetch 已有搜索结果中的百科页面

### 策略 3：Evidence over intuition — 证据优先原则

**问题**：LLM 有强烈的"内部知识自信"，会用自己的训练数据否定搜索结果中的正确证据。

**典型失败**：问题问"哪种作物的副产物用于制味精"，搜索结果指向"小麦"，但 LLM 认为"玉米淀粉制味精更常见"，切换到错误答案。

**设计**：
- 答案必须来自搜索结果，不得凭内部知识猜测
- **禁止用内部知识否定搜索证据**：如果搜索确认了 A，不得因"B 更常见"而切换
- **禁止用内部知识排除候选**：排除任何候选前必须搜索 `<候选> <约束>` 验证
- **Elimination checkpoint**：丢弃候选前必须在 Thought 中写"Elimination requires search: X Y — searching now"
- 搜索引擎排名靠前的结果（[1]-[3]）即使 snippet 关键词不完全匹配也应优先关注
- 工业流程/原料/供应链等领域知识明确标注为"不可信"，必须搜索验证

### 策略 4：Cross-validate — 多源交叉验证

**设计**：多个独立来源一致时置信度更高；来源冲突时继续搜索而非凭直觉选择。

### 策略 5：Challenge your hypothesis — 证伪搜索

**问题**：LLM 找到第一个看似合理的候选后倾向于立刻输出，缺少验证；或在验证阶段因"知名度""排序位置"等软信号随意更换答案。

**典型失败**：正确答案是 Harald Hauswald，但 LLM 在证伪阶段看到另一个更"知名"的摄影师 Bergemann，反复在两者之间摇摆。

**设计**：
- 找到候选后必须搜索至少一个替代方案
- **换答案标准**：只有新候选满足了原答案无法满足的**具体约束**时才允许切换
- **软信号不构成换答案理由**：排序偏见、知名度、象征关联、出现频率等均不可作为依据
- 等价候选（满足相同约束集）保持原答案不变
- 未被确认的约束 = 候选可能错误 → 立即放弃候选名，用未满足的约束重新搜索

### 策略 6：Multi-hop reasoning — 链式推理

**设计**：许多问题需要 A→B→C 的链式推理。每确认一个中间事实立即用 `Finding:` 记录，确保链不断裂。最终答案必须回答原始问题，而非中间步骤。

### 策略 7：When to fetch — 按需抓取

**设计**：
- 仅在搜索 snippet 过于模糊或截断时使用 fetch
- URL 必须来自之前的搜索结果
- 百科路由：中文题优先百度百科，其次维基百科；英文题优先 Wikipedia

### 策略 8：When and how to finish — 完成条件

**设计**：
- 所有子问题都有 Finding 确认，最终答案逻辑推导成立时才可 finish
- Finish 前必须：逐条复查约束 → 搜索排除当前候选的替代方案 → 检查答案语言
- 搜索预算 ≤ 2 时可跳过证伪

### 策略 9：Answer format — 答案格式

**问题**：精确匹配评测下，答案多一个字少一个字都算错。

**典型失败**：答"蒙达多利"而非"阿诺尔多·蒙达多利出版社"；答 "Desert Storm" 而非 "Operation Desert Shield and Desert Storm"。

**设计**：
- 使用权威来源中的**完整正式名称**（百科/官网原文）
- 题目未要求翻译时使用**原文名**（如德文书名 "Wie Baut Amerika?" 不译为英文）
- 多实体答案列出所有相关实体
- 外国人名中文译名必须从搜索结果中逐字复制，**禁止自造音译**
- 严格遵循题目声明的格式（分隔符、数字格式、全称/简称等）

### Prompt 层 vs 代码层的分工

| 职责 | Prompt 层 | 代码层 |
|------|----------|--------|
| **搜索什么** | 9 条策略引导搜索方向和多样性 | 精确重复拦截、过长查询压缩 |
| **何时完成** | 全约束检查规则 | 三阶段验证状态机、放弃检测 |
| **答案质量** | 格式规则、证据优先原则 | `format_answer()` 清理输出杂质 |
| **答案稳定性** | 换答案标准（具体约束差异） | `answer_switches` 计数器 + BLOCK 机制 |
| **记忆管理** | `Finding:` 语法引导 LLM 记录 | Findings 字典持久化 + Trace 智能压缩 |
| **幻觉防御** | — | `fabricated` 标志检测 + 警告注入 |

**设计思路**：**Prompt 负责"做对的事"，代码负责"不出错"。** Prompt 引导 LLM 正确推理，代码层在 LLM 偏离时兜底。两者缺一不可——纯靠 Prompt 无法阻止幻觉和无限摇摆，纯靠代码无法教会 LLM 搜索策略。

## Getting Started

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

可选 PDF 抽取：

```bash
pip install pypdf
```

### 2. 配置环境变量

在项目根目录创建 `.env`：

```dotenv
# 必需
BASE_URL=https://your-llm-endpoint.example.com/v1
API_KEY=sk-xxxxxxxx
LLM_MODEL=qwen3.5-plus
SERPER_API_KEY=serper_xxxxxxxx

# 可选（以下为默认值）
TOTAL_TIMEOUT=600        # 单题总超时（秒）
LLM_TIMEOUT=180          # 单次 LLM 调用超时（秒）
MAX_ITERATIONS=50        # ReAct 循环上限
MAX_SEARCH_QUERIES=50    # 每题搜索预算
MAX_RESULTS_PER_QUERY=8  # 单次搜索结果数
MAX_FETCH_PAGES=6        # 每题最多抓取页面数
FETCH_TIMEOUT=15         # 单页抓取超时（秒）
MAX_PAGE_CHARS=15000     # 单页送入 LLM 的最大字符数
```

### 3. 启动服务

```bash
uvicorn agent:app --host 0.0.0.0 --port 8000
```

## Testing

### 验证集测试

```bash
python scripts/test_validation.py          # 全部 11 题
python scripts/test_validation.py 3        # 只跑第 3 题
set DEBUG=1 && python scripts/test_validation.py 0  # Windows 开启调试
```

### 全量测试

```bash
python scripts/test_full.py               # 全部（已完成的自动跳过）
python scripts/test_full.py 5             # 只跑 id=5
python scripts/test_full.py 0-9           # 跑 id 0~9
python scripts/test_full.py --clean       # 清除进度重跑
python scripts/test_full.py --rerun 3,7   # 重跑指定 id
```

- 测试串行执行，避免触发 API 限流
- 单题耗时约 30s ~ 600s
- 日志：`log_validation.txt`（验证集）/ `log.txt`（全量）
- 结果：`sample/result.jsonl`

## API Reference

### `POST /`

标准问答接口，SSE 流式返回。

**请求**：

```json
{"question": "复杂的多跳问题..."}
```

**响应**（`text/event-stream`）：

```text
event: Ping

event: Message
data: {"answer": "最终"}

event: Message
data: {"answer": "答案"}
```

- `Ping`：长耗时保活（Agent 推理中）
- `Message`：答案片段，按序拼接即为完整答案

**示例**：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "Where is the capital of France?"}' \
  "http://127.0.0.1:8000/"
```

### `POST /ag-ui`

AG-UI 协议兼容接口，事件流格式略有不同。一般场景使用 `POST /` 即可。

## Project Structure

```text
/
├── agent.py              # FastAPI 入口，POST / 和 POST /ag-ui
├── agent_loop.py         # 桥接层：run_agent = run_react_agent
├── config.py             # 环境变量、LLM HTTP 调用、重试逻辑
├── react_agent.py        # ReAct 循环核心：状态机、验证、回退提取
├── react_prompt.py       # 系统 Prompt（9 条策略）
├── search_provider.py    # Serper 搜索封装、缓存、去重
├── page_fetcher.py       # Jina Reader + aiohttp 回退 + PDF 支持
├── utils.py              # 问题解析、答案格式清洗
├── requirements.txt
├── scripts/
│   ├── test_validation.py
│   └── test_full.py
└── sample/
    ├── validation.jsonl   # 验证集
    └── question.jsonl     # 正式题目
```

## License

MIT
