# Agent Deployment Guide

它基于 FastAPI 提供服务入口，内部采用单循环 ReAct Agent，通过搜索与页面抓取完成复杂检索问答，并通过 SSE 向客户端流式返回结果。

## 目录结构

```text
./
├── agent.py                # FastAPI 服务入口，提供 POST / 和 POST /ag-ui
├── agent_loop.py           # 统一暴露 run_agent()
├── config.py               # 环境变量加载、LLM HTTP 调用、日志配置
├── react_agent.py          # ReAct Agent 主流程
├── react_prompt.py         # Agent 提示词
├── search_provider.py      # Serper 搜索封装
├── page_fetcher.py         # Jina Reader + 直连抓取
├── utils.py                # 问题解析、清洗、答案格式化
├── requirements.txt        # Python 依赖
├── README.md
├── scripts/
│   ├── test_validation.py  # 验证集测试（11 题，含答案比对）
│   └── test_full.py        # 全量测试（100 题，输出结果 JSONL）
└── sample/
    ├── validation.jsonl     # 验证集（11 题，含答案）
    └── question.jsonl       # 正式题目（100 题，无答案）
```

## 设计架构

### 核心特点（相对纯 ReAct 的改进）

| 特点 | 纯 ReAct | 本系统 |
|------|---------|--------|
| **答案验证** | LLM 自行决定何时输出答案 | 三阶段证伪状态机：首次 finish → 强制证伪搜索 → 最终确认，防止未经验证的答案直接输出 |
| **中间记忆** | 仅靠 Trace 上下文窗口 | Findings 机制：`Finding: key = val` 结构化记录已确认事实，始终注入 prompt，不受 trace 压缩影响 |
| **幻觉防御** | 无 | 检测 LLM 自造 Observation/Result 内容，自动截断并注入警告；幻觉输出中的 finish 不进入候选池 |
| **答案稳定性** | LLM 可任意切换答案 | 换答案门槛控制：前 2 次允许切换（重置验证），第 3 次起 BLOCK；连续 3 次 BLOCK 同一答案触发死循环逃逸 |
| **超时容错** | 超时即丢失 | 全程累积 `answer_candidates`，超时后四阶段评分（finish 候选 → findings → LLM 总结 → 正则扫描）提取最优答案 |
| **Trace 管理** | 全量保留或固定窗口 | 智能压缩：近 5 步完整保留，旧步骤仅保留 Thought/Action、省略冗长 Observation，关键事实已在 Findings 中 |
| **搜索质量** | LLM 自由生成查询 | 精确重复拦截 + prompt 引导搜索多样性（维度轮换清单）+ 过长查询 LLM 自动压缩 + 跨查询 URL/snippet 去重 |
| **搜索结果处理** | 直接返回 | 保留 Google 原始排序 + 来源类型标签（Knowledge Graph / Answer Box / Encyclopedia）+ 百科 fetch 提示 |

### 总览

系统采用 **单循环 ReAct（Reason + Act）架构**：LLM 作为决策核心，在每一轮中进行推理（Thought）、选择动作（Action）、获取观察结果（Observation），循环往复直到得出最终答案。整个过程由代码驱动循环、LLM 驱动决策，二者协作完成复杂检索问答。

### 请求链路

```text
客户端 POST {"question": "..."} ─┐
                                  ▼
                         ┌─ agent.py ──────────────────────┐
                         │  FastAPI 路由解析请求体           │
                         │  启动异步 task 执行 Agent         │
                         │  等待期间每 5s 发送 SSE Ping 保活  │
                         └──────────┬───────────────────────┘
                                    ▼
                         ┌─ agent_loop.py ─────────────────┐
                         │  桥接层：run_agent = run_react_agent │
                         └──────────┬───────────────────────┘
                                    ▼
                         ┌─ react_agent.py ────────────────┐
                         │                                  │
                         │  1. 问题解析与状态初始化           │
                         │  2. ReAct 推理循环（≤30 轮）      │
                         │  3. 回退答案提取（若循环未 finish） │
                         │  4. 答案格式清洗                   │
                         │                                  │
                         └──────────┬───────────────────────┘
                                    ▼
                         ┌─ agent.py ──────────────────────┐
                         │  将答案拆分为 SSE Message 块       │
                         │  流式返回给客户端                  │
                         └─────────────────────────────────┘
```

### 完整推理流程

一个问题从输入到最终答案输出，经历以下四个阶段：

#### 阶段一：问题解析与状态初始化

`run_react_agent()` 首先调用 `parse_question()` 分析问题文本：

1. **语言检测**：统计中文字符比例，>30% 判定为 `zh`，否则为 `en`
2. **答案类型推断**：匹配 "多少"/"how many" 等模式判定 `answer_kind`（`number` / `entity`）
3. **格式提示提取**：检测 "全称"/"简称" 等关键词，以及 "格式形如：XXX" 等显式格式示例

解析结果初始化 `ReActState`，包含：问题文本、语言、答案类型、搜索/抓取/时间预算、trace 记录列表、findings 字典等。

#### 阶段二：ReAct 推理循环

系统核心。循环最多执行 `MAX_ITERATIONS`（默认 30）轮，每轮流程：

1. **守卫检查**：剩余时间 < 15s 或预算耗尽 → 强制退出
2. **组装 Prompt**：`_build_prompt()` 注入原始问题、Findings、历史 Trace（近 5 步完整，旧步骤 Observation 省略）、剩余预算
3. **调用 LLM**：`call_llm(prompt, temperature=0, timeout=动态超时)`，失败重试最多 3 次，连续 3 次错误退出
4. **解析输出**：`_parse_react_output()` 提取 Thought / Action / Action Input / Finding / fabricated 标志
5. **执行 Action**（见下方）
6. **追加 Trace**：本轮 Thought + Action + Observation 记入 `state.trace`

##### 工具调度

**`search(query)`** — 预处理（解析 `[en]`/`[zh]` 语言前缀、过长查询 LLM 压缩、精确重复拦截）→ `SerperProvider.search()`（自动语言检测、LRU+TTL 缓存、重试 2 次）→ `_format_search_results()`（保留原始排序、跨查询去重、来源类型标签、百科 fetch 提示）

**`fetch(url)`** — 守卫检查（URL 合法性、抓取预算、剩余时间 > 30s）→ `fetch_page_content()`（Jina Reader 优先 → aiohttp 回退 → PDF 用 pypdf）→ 截断至 `MAX_PAGE_CHARS`

**`finish(answer)`** — 最复杂的 Action，经过三层处理：

1. **放弃检测**：答案为空 / >200 字符 / 匹配放弃模式 → 若预算 > 5 则拒绝，要求继续搜索
2. **验证状态机**（三阶段）：
   - Phase 0→1：首次 finish，注入 FALSIFICATION CHECK（逐条检查约束、扫描未调查实体、搜索替代答案）
   - Phase 1→2：至少 2 次有效搜索后，注入 FINAL CHECK（审查证伪结果，禁止基于软信号换答案）
   - Phase 2+：接受答案，退出循环
3. **换答案控制**：前 2 次允许切换（重置到 Phase 0），第 3 次起 BLOCK（保留原答案继续验证），连续 3 次 BLOCK 同一答案 → 死循环逃逸

##### 辅助机制

| 机制 | 说明 |
|------|------|
| **Findings 记录** | LLM 输出 `Finding: key = val` 时自动记入 `state.findings`，后续每轮 prompt 始终注入，不受 trace 精简影响 |
| **Trace 精简** | 近 5 步保留完整 Observation，旧步骤超 400 字符的 Observation 替换为省略提示，Thought/Action 始终完整 |
| **幻觉检测** | 检测 LLM 自造 Observation/Result 内容，截断并标记 `fabricated`，下轮注入警告 |
| **动态 LLM 超时** | `min(LLM_TIMEOUT, max(30, 剩余时间 × 0.4))`，防止单次调用耗尽总时间 |
| **搜索语言自动检测** | 中文字符 >30% 时使用 `hl=zh-cn, gl=cn`，否则 `hl=en, gl=us`；LLM 也可通过 `[en]`/`[zh]` 前缀显式指定 |

#### 阶段三：回退答案提取

若循环结束时没有通过 `finish` 得到答案（超时/预算耗尽/连续错误），`_extract_best_answer_from_trace()` 按优先级从多个来源评分选最优：

1. **finish 提议历史**（`answer_candidates`）：基础分 0.90+，越晚提出微量加分
2. **Findings 字典**：优先匹配 key 含"答案"/"answer"的条目，基础分 0.75
3. **LLM 回退总结**（仅当最高分 < 0.85）：将最近 8 步 trace + findings 交给 LLM 提取，基础分 0.70
4. **正则扫描 Thought**（仅当前 3 源全空）：匹配"答案是 X"/"answer is X"/加粗文本，基础分 0.50

所有候选按 `answer_kind` 做 ±0.1 软调整（数字题偏好含数字的候选），选最高分。

#### 阶段四：答案格式清洗

`format_answer()` 对 LLM 输出做最小化清洗：

1. 取第一行
2. 去除常见前缀（"答案是"、"The answer is" 等）
3. 去除包裹引号（中英文引号、书名号）
4. 去除尾部中文句号
5. 去除尾部括注（如 "玉米 (Corn/Maize)" → "玉米"）

### 模块职责

- `agent.py`
  - 对外暴露全局 `app`
  - 提供根接口 `POST /`
  - 提供 AG-UI 兼容接口 `POST /ag-ui`
  - 在长耗时期间周期性发送 `event: Ping`
  - 将最终答案拆成一个或多个 `event: Message`

- `agent_loop.py`
  - 将服务入口与具体 Agent 实现解耦
  - 统一暴露 `run_agent(question)`

- `react_agent.py`
  - 负责单循环 ReAct 式推理
  - 驱动搜索、抓取、观察、结论生成

- `search_provider.py`
  - 调用 Serper 搜索接口
  - 对搜索结果做标准化与简单缓存

- `page_fetcher.py`
  - 优先通过 Jina Reader 获取网页正文
  - 失败时回退到直接抓取网页
  - 可选支持 PDF 文本抽取

- `config.py`
  - 从 `.env` 读取配置
  - 通过兼容 Chat Completions 的 HTTP 接口调用 LLM
  - 提供日志初始化与运行时限制参数

## 依赖说明

`requirements.txt` 中包含以下基础依赖：

- `fastapi`：服务框架
- `aiohttp`：异步搜索与抓取
- `python-dotenv`：环境变量加载
- `requests`：LLM HTTP 请求

### 可选依赖

- `pypdf`
  - 不在默认依赖中
  - 安装后可提升 PDF 页面文本抽取能力


## 环境变量

`config.py` 默认从当前目录下的 `.env` 加载配置。

### 必需项

- `BASE_URL`：兼容 Chat Completions 的服务根地址
- `API_KEY`：LLM 服务密钥
- `LLM_MODEL`：模型名
- `SERPER_API_KEY`：[serper.dev](https://serper.dev) 搜索密钥

### 常用可选项

- `TOTAL_TIMEOUT`：单题总超时，默认 `600`
- `LLM_TIMEOUT`：单次 LLM 调用超时，默认 `180`
- `MAX_ITERATIONS`：最大推理轮数，默认 `30`
- `MAX_SEARCH_QUERIES`：最大搜索次数，默认 `25`
- `MAX_RESULTS_PER_QUERY`：单次搜索结果数，默认 `8`
- `MAX_FETCH_PAGES`：单题最大抓取页面数，默认 `6`
- `FETCH_TIMEOUT`：单页面抓取超时，默认 `15`
- `MAX_PAGE_CHARS`：单页面送入模型的最大字符数，默认 `15000`
- `DEBUG`：是否输出详细日志，`1` 为开启
- `NO_TIMEOUT`：是否关闭总超时，`1` 为开启

### `.env` 示例

```dotenv
BASE_URL=https://your-llm-endpoint.example.com/v1
API_KEY=sk-xxxxxxxx
LLM_MODEL=qwen3-max
SERPER_API_KEY=serper_xxxxxxxx

TOTAL_TIMEOUT=600
LLM_TIMEOUT=180
MAX_ITERATIONS=30
MAX_SEARCH_QUERIES=25
MAX_RESULTS_PER_QUERY=8
MAX_FETCH_PAGES=6
FETCH_TIMEOUT=15
MAX_PAGE_CHARS=15000
DEBUG=0
NO_TIMEOUT=0
```

建议将真实密钥配置到部署平台环境变量中，不要把真实密钥提交到仓库。

## 完整复现步骤

下面的流程以“把 `/` 当作独立部署包”作为前提。

### 1. 获取代码

如果你准备单独部署，可直接复制本目录中的以下文件作为部署包根目录：

- `agent.py`
- `agent_loop.py`
- `config.py`
- `react_agent.py`
- `react_prompt.py`
- `search_provider.py`
- `page_fetcher.py`
- `utils.py`
- `requirements.txt`
- `scripts/test_validation.py`
- `scripts/test_full.py`

### 2. 创建虚拟环境并安装依赖

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

如果需要 PDF 抽取能力，可额外安装：

```bash
pip install pypdf
```

### 3. 配置 `.env`

在项目根目录下创建 `.env`：

```dotenv
BASE_URL=https://your-llm-endpoint.example.com/v1
API_KEY=sk-xxxxxxxx
LLM_MODEL=qwen3-max
SERPER_API_KEY=serper_xxxxxxxx
```

### 4. 先做语法检查

```bash
python -m py_compile agent.py agent_loop.py config.py page_fetcher.py react_agent.py search_provider.py scripts/test_full.py scripts/test_validation.py
```

### 5. 本地启动服务

在项目根目录内启动：

```bash
uvicorn agent:app --host 0.0.0.0 --port 8000
```

## 测试脚本

`scripts/` 目录下提供两个测试脚本，用于本地验证 Agent 的答题能力。脚本会自动加载项目根目录的 `.env` 并引入项目模块，从项目根目录运行即可。

### test_validation.py — 验证集测试

对 `sample/validation.jsonl`（11 题，含标准答案）逐题运行 Agent，归一化后精确匹配判定正误，输出正确率统计。

```bash
# 跑全部 11 题
python scripts/test_validation.py

# 只跑第 3 题（0-based 索引）
python scripts/test_validation.py 3

# 开启 DEBUG 日志（Windows）
set DEBUG=1 && python scripts/test_validation.py 0
```

日志输出到项目根目录 `log_validation.txt`。

### test_full.py — 全量测试

对 `sample/question.jsonl`（100 题，无答案）逐题运行 Agent，结果写入 `sample/result.jsonl`（JSONL 格式：`{"id": 0, "answer": "..."}`）。支持断点续跑。

```bash
# 跑全部 100 题（已完成的自动跳过）
python scripts/test_full.py

# 只跑 id=5 的题
python scripts/test_full.py 5

# 跑 id 0~9 的题
python scripts/test_full.py 0-9

# 清除进度，从头开始
python scripts/test_full.py --clean

# 重跑指定 id 的题
python scripts/test_full.py --rerun 3,7
```

日志输出到项目根目录 `log.txt`，结果文件为 `sample/result.jsonl`。

### 注意事项

- 测试脚本串行执行，避免并发触发第三方 API 限流
- 单题耗时约 30s ~ 600s（取决于搜索轮次和 LLM 响应速度）
- 个别题目因搜索引擎返回不了有效结果而失败属正常现象，关注整体正确率

## 调用方式

### 标准 EAS / LangStudio 风格请求

根接口要求：

- 方法：`POST /`
- 请求体：`{"question": "..."}`
- `Content-Type: application/json`
- 推荐 `Accept: text/event-stream`
- 支持客户端携带 `Authorization: Bearer <your_token>`

调用示例：

```bash
curl -X POST \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "Where is the capital of France?"}' \
  "http://127.0.0.1:8000/"
```

### 返回格式

返回类型为 `text/event-stream`。

长耗时阶段会先返回 `Ping` 保活事件；最终答案以一个或多个 `Message` 事件返回：

```text
event: Ping

event: Ping

event: Message
data: {"answer": "The"}

event: Message
data: {"answer": " capital"}

event: Message
data: {"answer": " is"}

event: Message
data: {"answer": " Paris"}
```

说明：

- `Ping` 事件用于防止长时间无数据导致网关断流
- `Message` 事件中的 `answer` 片段按顺序拼接后，即为最终答案
- 当前服务本身不校验 Bearer Token，通常由上游网关或平台负责鉴权

## AG-UI 兼容接口

除了根接口外，还保留了：

- `POST /ag-ui`

该接口用于兼容 AG-UI 风格请求与事件流格式；如果你的客户端只需要标准题目输入，优先使用 `POST /`。

## LangStudio / EAS 部署要点

如果要把当前目录作为代码模式部署包上传，建议遵守以下约束：

- 包根目录必须包含 `agent.py`
- `agent.py` 必须导出全局 `app`
- 服务入口是 FastAPI 应用，不需要额外包装
- 根接口协议为 `POST /` + SSE
- 已内置 `Ping` 保活逻辑，无需额外补 180 秒防断流代码
- 需要为运行环境提供外网访问，以访问 LLM 与 Serper

## 快速命令清单

安装依赖：

```bash
pip install -r requirements.txt
```

本地启动：

```bash
uvicorn agent:app --host 0.0.0.0 --port 8000
```

调用接口：

```bash
curl -X POST -H "Content-Type: application/json" -H "Accept: text/event-stream" -d '{"question":"Where is the capital of France?"}' "http://127.0.0.1:8000/"
```

