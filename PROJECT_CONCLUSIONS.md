# PaperForge 项目结论（2026-03-06）

## 1) 这个项目是做什么的

PaperForge 是一个面向学术论文生产的 LLM 自动化流水线，覆盖：

- 选题/idea 生成
- 实验执行与结果汇总
- 论文写作（LaTeX）
- 自动评审与迭代改写
- 云端（SSH）训练与结果回灌

它不是单纯“写文案”，而是“写作 + 实验 + 评估 + 回路优化”的完整系统。

## 2) 已修复的关键路径问题

此前项目内存在 `results/...` 与 `workspace/results/...` 的混用，容易导致：

- 命令按 README 运行却找不到目录
- 不同入口脚本产物位置不一致

本次已统一为从仓库根目录使用 `workspace/results/...`。

## 3) 最短本机跑通（从 0 到第一版 PDF）

```powershell
cd E:\pythonProjects\PaperForge

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 至少配置一个可用模型（示例：OpenAI）
$env:OPENAI_API_KEY="你的KEY"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENALEX_MAIL_ADDRESS="你的邮箱@example.com"

# 可选：预检
python -m engine.preflight --workspace .\workspace

# 生成首版草稿
python launch_mvp_workflow.py --phase bootstrap --experiment paper_writer --engine openalex --writeup-model gpt-4o-mini
```

首版 PDF 位置：

`workspace/results/paper_writer/<timestamp_idea>/paper_mvp_draft.pdf`

## 4) 各 phase 的主要落盘位置

假设工作区为：`workspace/results/paper_writer/<timestamp_idea>/`

- `bootstrap`
  - `run_1/`, 图表 PNG, `paper_mvp_draft.pdf`, `workflow_state.json`, `notes.txt`
- `feedback`
  - 读取 `uploads/`，归档到 `artifacts/user_*`，输出 `paper_with_feedback.pdf`
- `optimize`
  - 继续新增 `run_n/`，更新图表，输出 `paper_after_optimize.pdf`
- `refine`
  - 深度润色，输出 `paper_refined.pdf`，可能产生 `latex/checkpoints/`
- `cloud`
  - 远程结果落地 `remote_results/`，同步后更新 `artifacts/upload_manifest.json`

