# RAG 评测集

给检索和答案链路一个可重复的基线，避免凭感觉调参。

## 数据格式

`eval/golden_qa.jsonl`，每行一个 JSON 对象：

| 字段 | 说明 |
|---|---|
| `id` | 唯一标识 |
| `question` | 用户问题 |
| `answer` | 参考答案 |
| `relevant_snippets` | 答案在原文中的短语；评测时按去空白后的子串匹配定位相关 chunk |

## 运行

在仓库根目录执行：

```bash
# 混合检索基线（不重排）
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5

# 加重排
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5 --rerank

# 输出 JSON 报告
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5 \
  --json-out eval/results/baseline.json
```

评测只使用本地 embedding 和可选的重排模型，不调用 Ollama。

默认语料为仓库根目录 PDF + `eval/data/` 下所有支持格式；也可以用 `--corpus` 指定一个或多个文件。

中文对比使用 `.env` 中的：

```env
EMBED_MODEL_NAME=BAAI/bge-base-zh-v1.5
EMBED_QUERY_PREFIX=为这个句子生成表示以用于检索相关文章：
```

## 当前基线（8 个文件，140 个 chunk，46 条问题）

语料：仓库根目录样例 PDF + `eval/data/` 下的 6 个 PPTX 和 1 个数据库课程设计 PDF。

| 配置 | Recall@5 | MRR@5 | nDCG@5 |
|---|---|---|---|
| 英文 embedding + 混合检索 | 0.2591 | 0.2054 | 0.2021 |
| 英文 embedding + 混合检索 + BGE 重排 | 0.8141 | 0.8848 | 0.7960 |
| 中文 BGE embedding + 混合检索 | 0.8149 | 0.8116 | 0.7680 |
| 中文 BGE embedding + 混合检索 + BGE 重排 | **0.8982** | 0.8822 | **0.8446** |

结论：

- 英文 embedding 在中文语料上的候选召回是主要瓶颈，Recall@5 只有 0.2591
- 换成中文 BGE 后，不加重排就已经达到 0.8149，接近“英文 + 重排”的水平
- 中文 BGE + 重排取得最高 Recall@5 和 nDCG@5；MRR 与“英文 + 重排”基本持平
- 当前最佳配置下，46 条问题中只有 `float-bits` 仍然 miss，说明它的相关 chunk 不在 top-10 候选内

## 答案级评测

答案评测会跑完整的检索 + 生成链路，需要本地 Ollama 可用：

```bash
# 先跑少量问题，避免 46 条全量耗时过长
PYTHONIOENCODING=utf-8 python -m eval.run_answer_eval \
  --limit 10 --model-choice ollama \
  --json-out eval/results/answer.json
```

指标：

- **Token F1**：生成答案与参考答案的词级 F1
- **Citation precision / recall**：解析回答中的 `[S1]` 引用，检查被引用来源是否覆盖 golden snippets
- **Refusal rate**：出现“无法回答/信息不足”等拒答的比例

同一批 10 条问题的 v1/v2 对比（本地 `qwen2.5:3b`）：

| 指标 | v1（原始 Prompt） | v2（Prompt v2 + 引用规范化） |
|---|---|---|
| Token F1 | 0.1241 | **0.3417** |
| Citation precision | 0.3333 | **0.8000** |
| Citation recall | 0.3500 | **0.7500** |
| Refusal rate | 0.0000 | 0.0000 |
| Citation adherence | 未统计 | **1.0000** |
| Retrieval hit rate | 未统计 | **1.0000** |
| Mean answer chars | 未统计 | **72.8** |

v2 做了三件事：Prompt 改为“先给结论、每条事实带 `[S1]`、无依据时拒答”，温度降到 0.2，并把 `S1` / `[来源：S1]` / `（S1）` 统一规范成 `[S1]`。

同一条 v2.1 Prompt 下的三方对比（同一批 10 条问题）：

| 模型 | Token F1 | Citation P | Citation R | Adherence | Retrieval Hit | Mean chars |
|---|---|---|---|---|---|---|
| `qwen2.5:3b`（本地） | **0.4041** | 0.80 | 0.70 | 1.0 | 1.0 | **24.0** |
| `deepseek-v4.1-flash` | 0.3109 | **0.85** | **0.95** | 1.0 | 1.0 | 105.2 |
| `glm-5.3-flash` | 0.2535 | 0.80 | 0.90 | 1.0 | 1.0 | 117.8 |

解读：

- **Token F1 偏爱短答案**：本地 3B 的回答只有 24 字左右，所以在词面 F1 上领先；云端模型回答更完整，反而被扣分。
- **引用覆盖以 DeepSeek 最好**（precision 0.85 / recall 0.95），GLM 其次，3B 最低。
- 人工阅读 10 条回答：DeepSeek 基本全部正确；GLM 多数正确；3B 在 `temp-repair`（把“总电源”答成“水温传感器”）和 `idle-causes`（复述示例内容）上仍有错误。
- 检索命中率三方都是 1.0，说明差距来自生成层，不是检索层。

因此不能只用 Token F1 选模型。如果目标是回答质量与引用可信度，DeepSeek 目前最好；如果必须本地离线，3B 仍需要 Prompt v2.2 和更强的 grounding。

评测清洗逻辑已修复：`<details>` 思考内容会整段移除，不再污染 F1、answer_chars、refusal 和 citation 统计。

### LLM correctness / faithfulness judge

Token F1 会偏爱短答案，所以可以再加一层 LLM 裁判：

```bash
PYTHONIOENCODING=utf-8 python -m eval.run_answer_eval \
  --limit 10 --model-choice deepseek --judge-model auto \
  --json-out eval/results/answer_judge.json
```

`--judge-model auto` 会交叉选择裁判模型，避免自我偏好：

- 候选 DeepSeek → 裁判 GLM
- 候选 GLM → 裁判 DeepSeek
- 候选 Ollama → 裁判 DeepSeek

裁判会返回：

- `correct` / `correctness_score`（1–5）
- `faithful` / `unsupported_claims`
- `citation_support`（每个 `[S#]` 是否被对应片段支持）
- `reason`

新增指标：

- `correctness_rate`、`mean_correctness_score`
- `faithfulness_rate`
- `unsupported_claim_count`
- `citation_support_rate`
- `judge_error_rate`

2 条问题的冒烟结果（DeepSeek 候选、GLM 裁判）：

| 指标 | 结果 |
|---|---|
| correctness_rate | 1.0000 |
| mean_correctness_score | 5.0000 |
| faithfulness_rate | 1.0000 |
| unsupported_claim_count | 0 |
| citation_support_rate | 1.0000 |
| judge_error_rate | 0.0000 |

同一批答案的 Token F1 只有 0.3045，但裁判认为两条都正确且忠实，说明 F1 确实不适合单独作为质量指标。

注意：LLM judge 不是绝对真值，可能有长度偏好、自我偏好和位置偏好；建议固定 rubric、抽样人工复核，必要时用两个裁判交叉验证。判分输入会发送到云端模型。

### 三方 judge 对比（10 条，fallback 已启用）

| 候选模型 | 裁判 | Correctness | Mean score | Faithfulness | Unsupported claims | Citation support | Judge error | Fallback |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5:3b` | DeepSeek（GLM fallback） | 0.5000 | 3.5000 | 0.7000 | 4 | 0.6667 | 0% | 20% |
| `deepseek-v4.1-flash` | GLM | **1.0000** | **5.0000** | **1.0000** | **0** | **1.0000** | **0%** | **0%** |
| `glm-5.3-flash` | DeepSeek（GLM fallback） | 1.0000 | 5.0000 | 1.0000 | 0 | 0.9375 | 0% | 60% |

对应的 Token F1 分别是 **0.4051**、0.2492、0.2533——3B 在词面 F1 上仍然最高，但 judge 只在 5/10 条上认为它正确，且出现 4 个 unsupported claims。

**Fallback 说明**：DeepSeek 主裁判在部分问题上没有返回可解析的 JSON，现已自动切换到 GLM 裁判，并把 `max_tokens` 从 800 提高到 3000。3B 轮 fallback 使用率 20%，GLM 轮 60%。GLM 轮的 fallback 是 GLM 自评，可能带自我偏好；DeepSeek 候选由 GLM 裁判完成了 10/10 且无 fallback，是三方中最可靠的判分。

最终结论：**DeepSeek > GLM > 3B**（回答正确性与忠实度），而不是 F1 给出的 3B 领先。若必须本地离线，3B 仍需要 Prompt v2.2 和更强的 grounding。

## 指标说明

- **Recall@k**：top-k 中命中的相关 chunk 占全部相关 chunk 的比例
- **MRR@k**：第一个相关 chunk 的倒数排名
- **nDCG@k**：按排名加权的相关性，越靠前的相关结果得分越高
- **Token F1**：答案与参考答案的 token 级 F1
- **Citation precision**：被引用来源中真正相关来源的比例
- **Citation recall**：相关来源中被回答引用的比例

## 已知局限

- 46 条问题覆盖 8 个文件；最佳配置下仍有 `float-bits` 等少数题目的相关 chunk 不在 top-10 候选内
- 答案评测目前使用词面 F1 和引用匹配，没有接入 LLM faithfulness judge
- 相关 chunk 通过答案片段匹配确定；片段应尽量短且唯一，避免跨 chunk 边界
- `eval/data/` 中的课件和 PDF 是本地评测语料，是否纳入版本控制需自行判断
