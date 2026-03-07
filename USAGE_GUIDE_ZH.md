# PaperForge 详细使用指南（Windows / PowerShell）

更新时间：2026-03-06

本指南分为两部分：

1. 配置部分（环境、API、远程、成本与预检）
2. 使用部分（全自动模式、分阶段模式、远程回灌）

> 说明：项目产物目录已统一为 `workspace/results/...`。

---

## 一、配置部分

### 1. 运行前依赖

- Python 3.10+
- LaTeX 工具：`pdflatex`、`bibtex`、`chktex`
- 一个可用的 LLM API（OpenAI 或 Anthropic）
- 文献检索建议配置 OpenAlex 邮箱（可选 S2 API）

### 2. 创建虚拟环境并安装

```powershell
cd E:\pythonProjects\PaperForge

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果遇到“系统 Python 运行被拒绝”，通常是虚拟环境未激活。

### 3. API 与检索环境变量

推荐先用 OpenAI 跑通：

```powershell
$env:OPENAI_API_KEY = "你的OpenAIKey"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENALEX_MAIL_ADDRESS = "你的邮箱@example.com"
```

可选：

```powershell
$env:S2_API_KEY = "你的SemanticScholarKey"
```

如果使用 Anthropic：

```powershell
$env:ANTHROPIC_API_KEY = "你的AnthropicKey"
$env:ANTHROPIC_BASE_URL = "https://api.anthropic.com"
```

### 4. 写作迭代与成本控制（推荐）

```powershell
$env:WRITEUP_CITE_ROUNDS = "4"
$env:WRITEUP_LATEX_FIX_ROUNDS = "2"
$env:WRITEUP_SECOND_REFINEMENT = "0"
```

建议：

- 先用 `gpt-4o-mini` 做低成本端到端验证
- `--num-ideas 1` 起步
- 先 `--skip-novelty-check`，确认主流程可跑再加复杂度

### 5. 预检（强烈建议）

```powershell
python -m engine.preflight --workspace .\workspace
```

可带远程配置：

```powershell
python -m engine.preflight --workspace .\workspace --remote-config .\remote.yaml
```

### 6. 统一入口（可选）

`launch_user_entry.py` 可以把协议、模型和子流程统一到一个命令里，且支持 `--dry-run` 查看实际执行命令：

```powershell
python launch_user_entry.py mvp --phase bootstrap --claude-protocol openai --dry-run
```

---

## 二、使用部分

PaperForge 有两种主用法：

- **全自动**：给主题后自动走 idea→实验→写作→评审
- **分阶段**：你在每一阶段插入素材与反馈（推荐）

### A. 全自动模式（scientist）

适用于先快速验证能力、自动生成一版草稿。

```powershell
python launch_scientist.py `
  --experiment paper_writer `
  --num-ideas 1 `
  --skip-novelty-check `
  --engine openalex `
  --model gpt-4o-mini `
  --idea-model gpt-4o-mini `
  --code-model gpt-4o-mini `
  --writeup-model gpt-4o-mini `
  --review-model gpt-4o-mini
```

输出目录示例：

`workspace/results/paper_writer/20260306_190000_xxx/`

### B. 分阶段模式（mvp，推荐）

适用于你要逐步投喂素材、控制质量与成本。

#### B1) Bootstrap：初始化 + 首版草稿

```powershell
python launch_mvp_workflow.py `
  --phase bootstrap `
  --experiment paper_writer `
  --engine openalex `
  --idea-name llm_retrieval `
  --title "LLM + Retrieval 研究主题" `
  --description "你的研究目标、假设、方法和约束" `
  --writeup-model gpt-4o-mini `
  --refresh-literature
```

产物：

- `paper_mvp_draft.pdf`
- `run_1/`（模板基线是 `run_0/`）
- `notes.txt`
- `uploads/` 与 `artifacts/`

#### B2) 上传你自己的素材

假设工作区路径为 `$WS`，放置方式如下：

- `$WS/uploads/notes.md`：核心文字素材（最重要）
- `$WS/uploads/figures/`：图表/PDF/截图
- `$WS/uploads/code/`：代码、日志、配置

#### B3) Feedback：吸收素材并改稿

```powershell
python launch_mvp_workflow.py `
  --phase feedback `
  --run-dir "$WS" `
  --engine openalex `
  --writeup-model gpt-4o-mini `
  --refresh-literature
```

产物：

- `paper_with_feedback.pdf`
- `artifacts/upload_manifest.json`
- `notes.txt` 自动更新

#### B4) Optimize：追加实验并更新论文

```powershell
python launch_mvp_workflow.py `
  --phase optimize `
  --run-dir "$WS" `
  --optimize-runs 2 `
  --writeup-model gpt-4o-mini
```

产物：

- 新增 `run_n/`
- `metrics_across_runs.png`
- `paper_after_optimize.pdf`

#### B5) Refine：深度润色

```powershell
python launch_mvp_workflow.py `
  --phase refine `
  --run-dir "$WS" `
  --refine-profile deep `
  --writeup-model gpt-4o-mini
```

产物：

- `paper_refined.pdf`
- 可能生成 `latex/checkpoints/`

#### B6) Radar：文献雷达（找论文/读论文/方法追踪）

```powershell
python launch_mvp_workflow.py `
  --phase radar `
  --run-dir "$WS" `
  --engine openalex `
  --radar-seed "CTA strategy" `
  --year-min 2010 `
  --year-max 2026 `
  --radar-max-topics 12 `
  --radar-per-topic 8 `
  --radar-max-papers 120 `
  --radar-recent-years 3
```

年份筛选建议使用 `--year-min/--year-max`。历史参数 `--literature-year-after/--literature-year-before` 仍兼容。

产物：

- `literature_radar_report.md`：扩展主题、方法分布、新增文献、下一步方法建议
- `artifacts/literature_radar/latest_snapshot.json`：最新快照
- `artifacts/literature_radar/history/snapshot_*.json`：历史快照（用于“定期更新”对比）
- `notes.txt` 自动更新 `Literature Radar` 区块

### C. 云端执行与回灌（可选）

#### C1) 配置 `remote.yaml`

从模板复制：

```powershell
Copy-Item .\remote.example.yaml .\remote.yaml
```

至少正确填写：

- `host`、`username`、`auth`
- `upload_paths`
- `train_command`
- `results_dir`

#### C2) 单独测试远程 runner

```powershell
python -m engine.remote_runner --config .\remote.yaml --download-dir .\remote_results
```

可选子模式：

- `--upload-only`
- `--exec-only`
- `--download-only`

#### C3) 在工作流里执行 cloud phase

```powershell
python launch_mvp_workflow.py `
  --phase cloud `
  --run-dir "$WS" `
  --remote-config .\remote.yaml
```

产物：

- `$WS/remote_results/`（如果使用远程）
- 同步后的 `uploads/`、`artifacts/upload_manifest.json`

---

## 三、常见问题与建议

### 1) 这个项目是“自动搜索网页写论文”吗？

不是通用网页爬虫。它主要走 OpenAlex / Semantic Scholar 学术检索接口；更偏“研究流程自动化”而非“纯网页写手”。

### 2) 我必须提供论文素材吗？

不必须。可以先自动跑出草稿；但高质量成稿通常需要你在 `uploads/` 提供真实素材、实验与图表。

### 3) 如何低风险起步？

- 先跑 `bootstrap`
- 确认 PDF 与目录正常
- 再逐步跑 `feedback` → `optimize` → `refine`

### 4) 如何定位最新工作区？

```powershell
$WS = Get-ChildItem .\workspace\results\paper_writer |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
```

---

## 四、关键路径速查

- 配置模板：`key.example.sh`
- 远程模板：`remote.example.yaml`
- 统一入口：`launch_user_entry.py`
- 全自动入口：`launch_scientist.py`
- 分阶段入口：`launch_mvp_workflow.py`
- 预检：`engine/preflight.py`
- 远程执行：`engine/remote_runner.py`
- 结果同步：`sync_cloud_results_to_uploads.py`
