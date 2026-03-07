from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_AIDER_OPENAI_PROTOCOL_CACHE: dict[tuple[str, str], str] = {}


def _in_virtualenv() -> bool:
    return bool(getattr(sys, "base_prefix", sys.prefix) != sys.prefix)


def _require_virtualenv(script_name: str) -> None:
    allow_system = os.getenv("PAPERFORGE_ALLOW_SYSTEM_PYTHON", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if allow_system or _in_virtualenv():
        return

    raise SystemExit(
        "环境错误: 检测到正在使用系统 Python 运行 PaperForge。\n"
        f"当前解释器: {sys.executable}\n"
        "请先激活虚拟环境后再运行，例如:\n"
        f"  source .venv311/bin/activate && python {script_name} --help\n"
        "如需强制跳过校验，可设置 PAPERFORGE_ALLOW_SYSTEM_PYTHON=1。"
    )


if __name__ == "__main__":
    _require_virtualenv("launch_mvp_workflow.py")

from engine.llm import AVAILABLE_LLMS
from engine.literature_radar import run_literature_radar
from engine.mvp_workflow import (
    append_upload_feedback_to_notes,
    create_workspace_from_template,
    ensure_upload_interface,
    ingest_user_uploads,
    initialize_notes,
    load_idea_metadata,
    load_workflow_state,
    next_run_index,
    refresh_notes_with_literature,
    refresh_notes_with_literature_radar,
    refresh_notes_with_run_feedback,
    run_experiment_once,
    run_plotting,
    save_workflow_state,
    write_idea_metadata,
)
def _build_aider_model(model_name: str):
    from aider.models import Model

    def _first_non_empty_env(*keys: str) -> str | None:
        def _is_placeholder(value: str) -> bool:
            normalized = value.strip().lower()
            return normalized.startswith("your_") or normalized.startswith("your-")

        for key in keys:
            value = os.getenv(key)
            if value and value.strip():
                cleaned = value.strip()
                if _is_placeholder(cleaned):
                    continue
                return cleaned
        return None

    def _with_openai_headers(m: Model) -> Model:
        if not getattr(m, "extra_params", None):
            m.extra_params = {}
        extra_headers = dict(m.extra_params.get("extra_headers", {}))
        extra_headers.setdefault("User-Agent", os.getenv("OPENAI_USER_AGENT", "curl/8.7.1"))
        anthropic_beta = os.getenv("PAPERFORGE_ANTHROPIC_BETA", "").strip()
        if anthropic_beta:
            # Optional Anthropic prompt-caching header over OpenAI-compatible endpoint.
            extra_headers.setdefault("anthropic-beta", anthropic_beta)
        m.extra_params["extra_headers"] = extra_headers

        api_key = _first_non_empty_env(
            "OPENAI_WRITEUP_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
        )
        if api_key:
            m.extra_params["api_key"] = api_key

        api_base = _first_non_empty_env(
            "OPENAI_API_BASE",
            "OPENAI_WRITEUP_BASE_URL",
            "OPENAI_BASE_URL",
            "ANTHROPIC_BASE_URL",
        )
        if api_base:
            api_base = api_base.strip().rstrip("/")
            if not api_base.endswith("/v1"):
                api_base = f"{api_base}/v1"
            m.extra_params["api_base"] = api_base

        # Some OpenAI-compatible gateways return empty completions when max_tokens is omitted.
        # Keep an explicit default for stability, with env override support.
        compat_max = 4096
        compat_max_raw = os.getenv("PAPERFORGE_OPENAI_COMPAT_MAX_TOKENS", "").strip()
        if compat_max_raw:
            try:
                parsed = int(compat_max_raw)
                if parsed > 0:
                    compat_max = parsed
            except ValueError:
                pass
        m.extra_params.setdefault("max_tokens", compat_max)
        return m

    def _openai_protocol_override() -> str:
        mode = os.getenv("PAPERFORGE_OPENAI_PROTOCOL", "auto").strip().lower()
        if mode in {"chat", "responses", "auto"}:
            return mode
        return "auto"

    def _is_legacy_chat_protocol_error(exc: Exception) -> bool:
        text = str(exc).lower()
        if "/v1/chat/completions" in text and "/v1/responses" in text:
            return True
        return "unsupported legacy protocol" in text and "responses" in text

    def _resolve_openai_protocol_for_model(openai_model_name: str) -> str:
        override = _openai_protocol_override()
        if override in {"chat", "responses"}:
            return override

        api_key = _first_non_empty_env(
            "OPENAI_WRITEUP_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
        )
        api_base = _first_non_empty_env(
            "OPENAI_API_BASE",
            "OPENAI_WRITEUP_BASE_URL",
            "OPENAI_BASE_URL",
            "ANTHROPIC_BASE_URL",
        )
        if not api_key or not api_base:
            return "chat"

        api_base = api_base.strip().rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

        cache_key = (api_base, openai_model_name)
        cached = _AIDER_OPENAI_PROTOCOL_CACHE.get(cache_key)
        if cached in {"chat", "responses"}:
            return cached

        try:
            from openai import OpenAI

            headers = {
                "User-Agent": os.getenv("OPENAI_USER_AGENT", "curl/8.7.1"),
            }
            anthropic_beta = os.getenv("PAPERFORGE_ANTHROPIC_BETA", "").strip()
            if anthropic_beta:
                headers["anthropic-beta"] = anthropic_beta

            probe_client = OpenAI(api_key=api_key, base_url=api_base, default_headers=headers)
            probe_client.chat.completions.create(
                model=openai_model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            protocol = "chat"
        except Exception as exc:
            protocol = "responses" if _is_legacy_chat_protocol_error(exc) else "chat"

        _AIDER_OPENAI_PROTOCOL_CACHE[cache_key] = protocol
        return protocol

    def _openai_model_ref(openai_model_name: str) -> str:
        protocol = _resolve_openai_protocol_for_model(openai_model_name)
        if protocol == "responses":
            return f"openai/responses/{openai_model_name}"
        return f"openai/{openai_model_name}"

    route_claude_via_openai = os.getenv("PAPERFORGE_CLAUDE_OPENAI_COMPAT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if model_name == "gpt-5.3-codex xhigh":
        return _with_openai_headers(Model(_openai_model_ref("gpt-5.3-codex-xhigh")))
    if model_name == "deepseek-coder-v2-0724":
        return Model("deepseek/deepseek-coder")
    if model_name == "deepseek-reasoner":
        return Model("deepseek/deepseek-reasoner")
    if model_name == "llama3.1-405b":
        return Model("openrouter/meta-llama/llama-3.1-405b-instruct")
    if model_name.startswith("claude-") and route_claude_via_openai:
        return _with_openai_headers(Model(_openai_model_ref(model_name)))
    if model_name.startswith("claude-"):
        return Model(f"anthropic/{model_name}")
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        return _with_openai_headers(Model(_openai_model_ref(model_name)))
    return Model(model_name)


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    backup = {}
    for key, value in overrides.items():
        backup[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _profile_env(profile: str) -> Dict[str, str]:
    if profile == "fast":
        return {
            "WRITEUP_CITE_ROUNDS": "2",
            "WRITEUP_LATEX_FIX_ROUNDS": "1",
            "WRITEUP_SECOND_REFINEMENT": "0",
        }
    if profile == "deep":
        return {
            "WRITEUP_CITE_ROUNDS": "6",
            "WRITEUP_LATEX_FIX_ROUNDS": "3",
            "WRITEUP_SECOND_REFINEMENT": "1",
        }
    return {}


def _update_state(workspace: Path, **updates) -> None:
    state = load_workflow_state(str(workspace))
    state.update(updates)
    save_workflow_state(str(workspace), state)


def _resolve_workspace(path_or_none: str | None) -> Path | None:
    if not path_or_none:
        return None
    path = Path(path_or_none).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _copy_phase_pdf(workspace: Path, idea_name: str, output_name: str) -> None:
    generated = workspace / f"{idea_name}.pdf"
    if generated.exists():
        shutil.copy2(generated, workspace / output_name)


def _run_writeup_phase(
    workspace: Path,
    writeup_model: str,
    engine: str,
    history_name: str,
    output_pdf_name: str,
    profile: str,
) -> None:
    from aider.coders import Coder
    from aider.io import InputOutput
    from engine.llm import create_client
    from engine.perform_writeup import perform_writeup

    idea = load_idea_metadata(str(workspace))
    notes = workspace / "notes.txt"
    writeup_file = workspace / "latex" / "template.tex"
    exp_file = workspace / "experiment.py"
    vis_file = workspace / "plot.py"

    fnames = [str(writeup_file), str(notes)]
    if exp_file.exists():
        fnames.insert(0, str(exp_file))
    if vis_file.exists():
        fnames.append(str(vis_file))

    io = InputOutput(
        yes=True,
        chat_history_file=str(workspace / f"{history_name}_writeup_aider.txt"),
    )
    edit_format = os.getenv("PAPERFORGE_AIDER_EDIT_FORMAT", "udiff").strip() or "udiff"
    coder = Coder.create(
        main_model=_build_aider_model(writeup_model),
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format=edit_format,
        auto_lint=False,
    )
    client, client_model = create_client(writeup_model)
    env_overrides = _profile_env(profile)
    with _temporary_env(env_overrides):
        perform_writeup(
            idea=idea,
            folder_name=str(workspace),
            coder=coder,
            cite_client=client,
            cite_model=client_model,
            engine=engine,
        )
    _copy_phase_pdf(workspace, idea_name=idea["Name"], output_name=output_pdf_name)


def _resolve_year_filters(args: argparse.Namespace) -> tuple[int | None, int | None]:
    year_min = args.year_min
    year_max = args.year_max
    legacy_min = args.literature_year_after
    legacy_max = args.literature_year_before

    if year_min is None and legacy_min is not None:
        year_min = legacy_min
    elif year_min is not None and legacy_min is not None and year_min != legacy_min:
        raise SystemExit(
            "Year filter conflict: --year-min and --literature-year-after are both set "
            f"with different values ({year_min} vs {legacy_min})."
        )

    if year_max is None and legacy_max is not None:
        year_max = legacy_max
    elif year_max is not None and legacy_max is not None and year_max != legacy_max:
        raise SystemExit(
            "Year filter conflict: --year-max and --literature-year-before are both set "
            f"with different values ({year_max} vs {legacy_max})."
        )

    if year_min is not None and year_max is not None and int(year_min) >= int(year_max):
        raise SystemExit(
            f"Invalid year range: --year-min ({year_min}) must be less than --year-max ({year_max})."
        )

    return year_min, year_max


def _phase_bootstrap(args: argparse.Namespace, workspace: Path | None) -> Path:
    if workspace is None:
        try:
            workspace = Path(
                create_workspace_from_template(args.experiment, args.idea_name)
            ).resolve()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))
    else:
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

    write_idea_metadata(
        str(workspace),
        idea_name=args.idea_name,
        title=args.title,
        description=args.description,
    )
    notes_path = Path(
        initialize_notes(
            str(workspace),
            title=args.title,
            description=args.description,
            overwrite=False,
        )
    )
    ensure_upload_interface(str(workspace))

    if args.refresh_literature:
        year_min, year_max = _resolve_year_filters(args)
        refresh_notes_with_literature(
            notes_path=str(notes_path),
            query=args.title,
            engine=args.engine,
            top_k=args.literature_top_k,
            year_before=year_max,
            year_after=year_min,
        )

    mvp_ok = False
    if not args.skip_mvp_run:
        run_info = run_experiment_once(
            workspace=str(workspace),
            python_bin=args.python_bin,
            run_index=args.bootstrap_run_index,
        )
        mvp_ok = run_info["returncode"] == 0
        print(
            "[bootstrap] run_{idx} returncode={rc}".format(
                idx=run_info["run_index"], rc=run_info["returncode"]
            )
        )
        if run_info["stderr"]:
            print(run_info["stderr"])
        plot_info = run_plotting(str(workspace), python_bin=args.python_bin)
        print(f"[bootstrap] plot returncode={plot_info['returncode']}")
        if plot_info["stderr"]:
            print(plot_info["stderr"])

    refresh_notes_with_run_feedback(str(workspace), str(notes_path))

    if not args.skip_writeup:
        _run_writeup_phase(
            workspace=workspace,
            writeup_model=args.writeup_model,
            engine=args.engine,
            history_name="mvp_bootstrap",
            output_pdf_name="paper_mvp_draft.pdf",
            profile="fast",
        )

    _update_state(
        workspace,
        phase="bootstrap_completed",
        mvp_completed=mvp_ok,
        upload_interface_ready=True,
    )
    print(f"[bootstrap] workspace={workspace}")
    return workspace


def _phase_feedback(args: argparse.Namespace, workspace: Path) -> None:
    ensure_upload_interface(str(workspace))
    manifest = ingest_user_uploads(str(workspace))
    notes_path = workspace / "notes.txt"
    if notes_path.exists():
        append_upload_feedback_to_notes(str(notes_path), manifest)
        refresh_notes_with_run_feedback(str(workspace), str(notes_path))

    if args.refresh_literature:
        year_min, year_max = _resolve_year_filters(args)
        refresh_notes_with_literature(
            notes_path=str(notes_path),
            query=args.title,
            engine=args.engine,
            top_k=args.literature_top_k,
            year_before=year_max,
            year_after=year_min,
        )

    if not args.skip_writeup:
        _run_writeup_phase(
            workspace=workspace,
            writeup_model=args.writeup_model,
            engine=args.engine,
            history_name="mvp_feedback",
            output_pdf_name="paper_with_feedback.pdf",
            profile="balanced",
        )

    _update_state(
        workspace,
        phase="feedback_completed",
        ingested_uploads=True,
        upload_manifest=str(workspace / "artifacts" / "upload_manifest.json"),
    )
    print(f"[feedback] workspace={workspace}")


def _phase_optimize(args: argparse.Namespace, workspace: Path) -> None:
    for _ in range(max(0, int(args.optimize_runs))):
        run_idx = next_run_index(str(workspace))
        run_info = run_experiment_once(
            workspace=str(workspace),
            python_bin=args.python_bin,
            run_index=run_idx,
        )
        print(
            "[optimize] run_{idx} returncode={rc}".format(
                idx=run_info["run_index"], rc=run_info["returncode"]
            )
        )
        if run_info["stderr"]:
            print(run_info["stderr"])
        if run_info["returncode"] != 0:
            break

    plot_info = run_plotting(str(workspace), python_bin=args.python_bin)
    print(f"[optimize] plot returncode={plot_info['returncode']}")
    if plot_info["stderr"]:
        print(plot_info["stderr"])

    notes_path = workspace / "notes.txt"
    refresh_notes_with_run_feedback(str(workspace), str(notes_path))

    if not args.skip_writeup:
        _run_writeup_phase(
            workspace=workspace,
            writeup_model=args.writeup_model,
            engine=args.engine,
            history_name="mvp_optimize",
            output_pdf_name="paper_after_optimize.pdf",
            profile="balanced",
        )

    _update_state(workspace, phase="optimize_completed")
    print(f"[optimize] workspace={workspace}")


def _phase_refine(args: argparse.Namespace, workspace: Path) -> None:
    if not args.skip_writeup:
        _run_writeup_phase(
            workspace=workspace,
            writeup_model=args.writeup_model,
            engine=args.engine,
            history_name="mvp_refine",
            output_pdf_name="paper_refined.pdf",
            profile=args.refine_profile,
        )

    _update_state(workspace, phase="refine_completed")
    print(f"[refine] workspace={workspace}")


def _phase_cloud(args: argparse.Namespace, workspace: Path) -> None:
    cmd = [
        args.python_bin,
        str(ROOT / "run_cloud_pipeline_cycle.py"),
        "--workspace",
        str(workspace),
    ]
    if args.cloud_run_dir:
        cmd += ["--cloud-run-dir", str(args.cloud_run_dir)]
    if args.pipeline_root:
        cmd += ["--pipeline-root", str(args.pipeline_root)]
    if args.pipeline_config:
        cmd += ["--config", args.pipeline_config]
    if args.pipeline_run_name:
        cmd += ["--run-name", args.pipeline_run_name]
    if args.pipeline_mode:
        cmd += ["--mode", args.pipeline_mode]
    if args.pipeline_hardware_profile:
        cmd += ["--hardware-profile", args.pipeline_hardware_profile]
    if args.pipeline_device:
        cmd += ["--device", args.pipeline_device]
    if args.remote_config:
        cmd += ["--remote-config", str(args.remote_config)]
    if args.cloud_skip_run:
        cmd.append("--skip-run")
    if args.cloud_skip_sync:
        cmd.append("--skip-sync")
    print("[cloud]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _resolve_radar_seed(args: argparse.Namespace, workspace: Path) -> str:
    explicit = (args.radar_seed or "").strip()
    if explicit:
        return explicit

    idea_meta_path = workspace / "workflow_idea.json"
    if idea_meta_path.exists():
        idea = load_idea_metadata(str(workspace))
        idea_title = (idea.get("Title") or "").strip()
        if idea_title:
            return idea_title

    title = (args.title or "").strip()
    default_title = "Iterative Academic Paper Writing with Upload Feedback"
    if title and title != default_title:
        return title

    raise SystemExit(
        "phase `radar` requires a meaningful seed topic. "
        "Please pass --radar-seed \"<your topic>\"."
    )


def _phase_radar(args: argparse.Namespace, workspace: Path) -> None:
    seed_topic = _resolve_radar_seed(args, workspace)
    year_min, year_max = _resolve_year_filters(args)

    try:
        summary = run_literature_radar(
            workspace=str(workspace),
            seed_topic=seed_topic,
            engine=args.engine,
            max_topics=args.radar_max_topics,
            per_topic=args.radar_per_topic,
            max_papers=args.radar_max_papers,
            year_before=year_max,
            year_after=year_min,
            recent_years=args.radar_recent_years,
        )
    except Exception as exc:
        raise SystemExit(f"[radar] failed: {exc}") from exc

    notes_path = workspace / "notes.txt"
    if notes_path.exists():
        try:
            refresh_notes_with_literature_radar(str(notes_path), summary)
        except Exception as exc:
            raise SystemExit(f"[radar] failed to update notes.txt: {exc}") from exc

    _update_state(
        workspace,
        phase="radar_completed",
        radar_report=summary.get("report_path"),
        radar_snapshot=summary.get("latest_snapshot_path"),
        radar_seed=seed_topic,
    )
    print(
        "[radar] papers={papers} new={new} report={report}".format(
            papers=summary.get("paper_count", 0),
            new=summary.get("new_paper_count", 0),
            report=summary.get("report_path", ""),
        )
    )


def _default_pipeline_root() -> str:
    return ""


def _check_latex_dependencies() -> bool:
    required = ["pdflatex", "bibtex", "chktex"]
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        print(
            "Error: Required LaTeX dependencies not found: " + ", ".join(missing),
            file=sys.stderr,
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MVP staged workflow launcher")
    p.add_argument(
        "--phase",
        choices=["bootstrap", "feedback", "optimize", "refine", "cloud", "radar", "all"],
        default="bootstrap",
    )
    p.add_argument("--experiment", default="paper_writer")
    p.add_argument("--run-dir", default=None, help="Existing workspace path")
    p.add_argument("--idea-name", default="paper_writer_mvp_pipeline")
    p.add_argument(
        "--title",
        default="Iterative Academic Paper Writing with Upload Feedback",
    )
    p.add_argument(
        "--description",
        default="Generate draft, ingest uploaded server outputs, and iteratively refine the paper.",
    )
    p.add_argument(
        "--engine",
        choices=["semanticscholar", "openalex"],
        default="openalex",
    )
    p.add_argument(
        "--writeup-model",
        default="claude-sonnet-4-6",
        choices=AVAILABLE_LLMS,
    )
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--skip-writeup", action="store_true")
    p.add_argument("--skip-mvp-run", action="store_true")
    p.add_argument("--bootstrap-run-index", type=int, default=1)
    p.add_argument("--optimize-runs", type=int, default=2)
    p.add_argument(
        "--refine-profile",
        choices=["fast", "balanced", "deep"],
        default="balanced",
    )
    p.add_argument("--refresh-literature", action="store_true")
    p.add_argument("--literature-top-k", type=int, default=5)
    p.add_argument(
        "--literature-year-before",
        type=int,
        default=None,
        help="[Compat] Only keep papers with year < this value. Prefer --year-max.",
    )
    p.add_argument(
        "--literature-year-after",
        type=int,
        default=None,
        help="[Compat] Only keep papers with year > this value. Prefer --year-min.",
    )
    p.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Keep papers with year > this value (preferred alias of --literature-year-after).",
    )
    p.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Keep papers with year < this value (preferred alias of --literature-year-before).",
    )
    p.add_argument(
        "--radar-seed",
        default=None,
        help="Seed keyword for literature radar expansion (defaults to --title).",
    )
    p.add_argument(
        "--radar-max-topics",
        type=int,
        default=12,
        help="Maximum number of expanded topics for radar search.",
    )
    p.add_argument(
        "--radar-per-topic",
        type=int,
        default=8,
        help="Number of papers fetched per expanded topic.",
    )
    p.add_argument(
        "--radar-max-papers",
        type=int,
        default=120,
        help="Maximum number of deduplicated papers kept in radar snapshot.",
    )
    p.add_argument(
        "--radar-recent-years",
        type=int,
        default=3,
        help="Recent-year window used for method trend extraction.",
    )

    p.add_argument("--run-cloud-cycle", action="store_true")
    p.add_argument(
        "--cloud-run-dir",
        default=None,
        help="Local directory containing server/cloud outputs to ingest.",
    )
    p.add_argument("--pipeline-root", default=_default_pipeline_root())
    p.add_argument("--pipeline-config", default=None)
    p.add_argument("--pipeline-run-name", default="final_full")
    p.add_argument("--pipeline-mode", choices=["auto", "real", "sim"], default="auto")
    p.add_argument("--pipeline-hardware-profile", default=None)
    p.add_argument("--pipeline-device", default=None)
    p.add_argument("--cloud-skip-run", action="store_true")
    p.add_argument("--cloud-skip-sync", action="store_true")
    p.add_argument("--remote-config", default=None, help="Path to remote.yaml for SSH remote execution in cloud phase.")
    return p


def _require_workspace(workspace: Path | None, phase: str) -> Path:
    if workspace is None:
        raise SystemExit(f"--run-dir is required for phase `{phase}`")
    if not workspace.exists():
        raise SystemExit(f"workspace not found: {workspace}")
    return workspace.resolve()


def main() -> None:
    args = build_parser().parse_args()
    if (
        not args.skip_writeup
        and args.phase in {"bootstrap", "feedback", "optimize", "refine", "all"}
        and not _check_latex_dependencies()
    ):
        raise SystemExit(1)
    workspace = _resolve_workspace(args.run_dir)

    if args.phase == "bootstrap":
        workspace = _phase_bootstrap(args, workspace)
    elif args.phase == "feedback":
        _phase_feedback(args, _require_workspace(workspace, args.phase))
    elif args.phase == "optimize":
        _phase_optimize(args, _require_workspace(workspace, args.phase))
    elif args.phase == "refine":
        _phase_refine(args, _require_workspace(workspace, args.phase))
    elif args.phase == "cloud":
        _phase_cloud(args, _require_workspace(workspace, args.phase))
    elif args.phase == "radar":
        _phase_radar(args, _require_workspace(workspace, args.phase))
    elif args.phase == "all":
        workspace = _phase_bootstrap(args, workspace)
        if args.run_cloud_cycle:
            _phase_cloud(args, workspace)
        _phase_feedback(args, workspace)
        _phase_optimize(args, workspace)
        _phase_refine(args, workspace)
        _phase_radar(args, workspace)
    else:
        raise SystemExit(f"Unsupported phase: {args.phase}")

    if workspace is not None:
        print(f"[done] workspace={workspace}")


if __name__ == "__main__":
    main()
