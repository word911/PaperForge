from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engine.generate_ideas import search_for_papers


_TOPIC_EXPANSION_MAP: Dict[str, List[str]] = {
    "cta": [
        "commodity trading advisor",
        "managed futures",
        "trend following futures",
        "time-series momentum futures",
        "regime switching trading",
        "volatility targeting futures",
        "risk parity managed futures",
        "crisis alpha managed futures",
    ],
    "commodity trading advisor": [
        "managed futures",
        "trend following",
        "time-series momentum",
        "regime switching",
        "hidden markov model trading",
    ],
    "managed futures": [
        "trend following futures",
        "time-series momentum futures",
        "carry strategy futures",
        "risk parity futures",
        "tail risk managed futures",
    ],
    "trend following": [
        "time-series momentum",
        "breakout strategy",
        "moving average trading rule",
        "trend following with volatility scaling",
    ],
    "regime switching": [
        "markov switching model trading",
        "hidden markov model finance",
        "state-space market regime detection",
        "dynamic allocation under regimes",
    ],
    "momentum": [
        "time-series momentum",
        "cross-asset momentum",
        "momentum crash",
        "tail risk in momentum",
    ],
    "状态切换": [
        "regime switching trading",
        "markov switching model finance",
        "hidden markov model trading",
    ],
    "趋势跟踪": [
        "trend following futures",
        "time-series momentum",
        "managed futures trend strategy",
    ],
    "动量策略": [
        "time-series momentum",
        "cross-asset momentum",
        "momentum with volatility targeting",
    ],
}


_THEME_KEYWORDS: Dict[str, List[str]] = {
    "Trend_Following": [
        "trend following",
        "trend-following",
        "moving average",
        "breakout",
    ],
    "Momentum": [
        "momentum",
        "time-series momentum",
        "time series momentum",
        "cross-sectional momentum",
    ],
    "Regime_Switching": [
        "regime switching",
        "markov switching",
        "hidden markov",
        "state-space",
        "state switching",
        "regime-aware",
    ],
    "Volatility_and_Tail_Risk": [
        "volatility targeting",
        "value at risk",
        "tail risk",
        "drawdown",
        "extreme value",
        "garch",
        "illiquidity",
    ],
    "Portfolio_and_Risk_Budgeting": [
        "risk parity",
        "portfolio",
        "allocation",
        "diversification",
        "semivariance",
        "carry trade",
    ],
    "Execution_and_Microstructure": [
        "transaction cost",
        "slippage",
        "market impact",
        "microstructure",
        "liquidity",
        "commitments of traders",
    ],
    "ML_and_Data_Driven": [
        "machine learning",
        "deep learning",
        "reinforcement learning",
        "random forest",
        "neural network",
        "feature selection",
    ],
}

_FINANCE_POSITIVE_KEYWORDS = [
    "commodity trading advisor",
    "managed futures",
    "futures",
    "trend following",
    "momentum",
    "trading",
    "portfolio",
    "risk parity",
    "carry",
    "hedge fund",
    "market regime",
    "markov switching",
    "volatility targeting",
]

_MEDICAL_NEGATIVE_KEYWORDS = [
    "coronary",
    "angiography",
    "anatomical",
    "disease",
    "patient",
    "clinical",
    "cardiac",
    "tumor",
    "therapy",
    "diagnosis",
]

MAX_TITLE_CHARS = 300
MAX_VENUE_CHARS = 180
MAX_AUTHORS_CHARS = 400
MAX_ABSTRACT_CHARS = 4000
MAX_QUERY_CHARS = 180
MAX_TITLE_KEY_CHARS = 200
MAX_METHOD_INSIGHTS = 8
MAX_MATCHED_QUERIES = 8


def _slugify(text: str, default: str = "topic") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or default


def _normalize_topic(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _normalize_title(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "", (text or "").lower())
    return normalized[:MAX_TITLE_KEY_CHARS]


def _sanitize_text(value: object, *, max_chars: int) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = " ".join(text.strip().split())
    return text[:max_chars]


def _extract_year(value: object) -> Optional[int]:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    max_year = datetime.now().year + 10
    if year < 1900 or year > max_year:
        return None
    return year


def _extract_citations(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _expand_topics(seed_topic: str, max_topics: int = 12) -> List[str]:
    seed = (seed_topic or "").strip()
    if not seed:
        return []

    normalized_seed = _normalize_topic(seed)
    topics: List[str] = [seed]
    seen = {_normalize_topic(seed)}

    for key, expansions in _TOPIC_EXPANSION_MAP.items():
        key_norm = _normalize_topic(key)
        if key_norm in normalized_seed or normalized_seed in key_norm:
            for item in expansions:
                item_norm = _normalize_topic(item)
                if item_norm and item_norm not in seen:
                    topics.append(item)
                    seen.add(item_norm)

    # Trading defaults when no family matched strongly.
    if len(topics) < 4:
        defaults = [
            "trend following",
            "time-series momentum",
            "regime switching",
            "volatility targeting",
            "risk parity",
        ]
        for item in defaults:
            item_norm = _normalize_topic(item)
            if item_norm not in seen:
                topics.append(item)
                seen.add(item_norm)

    return topics[: max(1, int(max_topics))]


def _detect_method_themes(title: str, abstract: str, venue: str) -> List[str]:
    haystack = f"{title} {abstract} {venue}".lower()
    themes: List[str] = []
    for theme, keywords in _THEME_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            themes.append(theme)
    if not themes:
        themes.append("General_Systematic_Trading")
    return themes


def _paper_key(title: str, year: Optional[int]) -> Tuple[str, int]:
    return _normalize_title(title), int(year or 0)


def _is_likely_finance_paper(seed_topic: str, title: str, abstract: str, venue: str) -> bool:
    seed = (seed_topic or "").lower()
    haystack = f"{title} {abstract} {venue}".lower()
    pos_hits = sum(1 for kw in _FINANCE_POSITIVE_KEYWORDS if kw in haystack)
    neg_hits = sum(1 for kw in _MEDICAL_NEGATIVE_KEYWORDS if kw in haystack)

    if neg_hits > 0 and pos_hits == 0:
        return False
    if "cta" in seed and neg_hits > 0 and "commodity trading advisor" not in haystack:
        return False

    # Keep papers with at least one finance-trading signal.
    return pos_hits > 0


def _load_previous_snapshot(latest_snapshot_path: Path) -> Dict:
    if not latest_snapshot_path.exists():
        return {}
    try:
        return json.loads(latest_snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=f".{path.name}.tmp.",
        ) as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _build_method_insights(
    seed_topic: str,
    theme_counts: Counter,
    recent_theme_counts: Counter,
    new_paper_count: int,
) -> List[str]:
    insights: List[str] = []
    seed = (seed_topic or "").strip()

    if recent_theme_counts.get("Regime_Switching", 0) and recent_theme_counts.get("Momentum", 0):
        insights.append(
            "优先推进“状态门控动量”路线：用 Regime-Switching 先做开关仓，再在状态内做动量权重分配。"
        )
    if recent_theme_counts.get("Trend_Following", 0) and recent_theme_counts.get(
        "Volatility_and_Tail_Risk", 0
    ):
        insights.append(
            "趋势信号建议与波动/尾部风控强绑定：趋势决定方向，波动与回撤状态决定杠杆倍数。"
        )
    if recent_theme_counts.get("Portfolio_and_Risk_Budgeting", 0):
        insights.append(
            "可增加组合层研究：从单信号优化转向“信号 + 风险预算 + 跨品种相关性约束”的联合优化。"
        )
    if recent_theme_counts.get("Execution_and_Microstructure", 0):
        insights.append(
            "把交易成本与流动性约束提前到研究主流程，避免只在最终回测阶段补做成本扣减。"
        )
    if recent_theme_counts.get("ML_and_Data_Driven", 0):
        insights.append(
            "可试“机器学习只做状态识别，交易规则保持可解释”方案，降低黑箱风险并提高可审计性。"
        )

    if theme_counts.get("Regime_Switching", 0) == 0:
        insights.append("当前样本中的状态切换主题偏弱，建议追加 Markov/HMM 相关定向检索。")
    if theme_counts.get("Momentum", 0) == 0:
        insights.append("当前样本中的动量主题偏弱，建议追加 time-series momentum 定向检索。")
    if new_paper_count == 0:
        insights.append("本轮未发现新增文献，建议扩大检索词或放宽年份过滤窗口。")
    else:
        insights.append(f"本轮识别到 {new_paper_count} 篇新增文献，可优先做摘要级精读与方法对照。")

    if seed:
        insights.append(f"以“{seed}”为主线时，建议每周固定刷新并保留快照，跟踪方法主题漂移。")

    deduped: List[str] = []
    seen = set()
    for item in insights:
        norm = _normalize_topic(item)
        if norm and norm not in seen:
            deduped.append(item)
            seen.add(norm)
    return deduped[:MAX_METHOD_INSIGHTS]


def _paper_to_md_line(idx: int, paper: Dict) -> str:
    title = paper.get("title", "Untitled")
    year = paper.get("year", "?")
    venue = paper.get("venue", "Unknown")
    cites = paper.get("citationCount", 0)
    themes = ", ".join(paper.get("method_themes", []))
    queries = ", ".join(paper.get("matched_queries", [])[:3])
    return f"{idx}. {title} ({year}, {venue}, cites={cites}) | themes: {themes} | via: {queries}"


def _render_markdown_report(
    seed_topic: str,
    engine: str,
    expanded_topics: List[str],
    papers: List[Dict],
    new_papers: List[Dict],
    method_insights: List[str],
    generated_at: str,
    recent_year_threshold: int,
) -> str:
    theme_counts = Counter()
    recent_theme_counts = Counter()
    for paper in papers:
        for theme in paper.get("method_themes", []):
            theme_counts[theme] += 1
            if (paper.get("year") or 0) >= recent_year_threshold:
                recent_theme_counts[theme] += 1

    lines: List[str] = []
    lines.append("# Literature Radar Report")
    lines.append("")
    lines.append(f"- Generated at: {generated_at}")
    lines.append(f"- Seed topic: `{seed_topic}`")
    lines.append(f"- Engine: `{engine}`")
    lines.append(f"- Expanded topics: {len(expanded_topics)}")
    lines.append(f"- Papers kept: {len(papers)}")
    lines.append(f"- New since previous run: {len(new_papers)}")
    lines.append("")

    lines.append("## Expanded Topics")
    for topic in expanded_topics:
        lines.append(f"- {topic}")
    lines.append("")

    lines.append("## Method Theme Coverage")
    lines.append("| Theme | Count | Recent Count |")
    lines.append("| --- | ---: | ---: |")
    for theme, count in sorted(theme_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {theme} | {count} | {recent_theme_counts.get(theme, 0)} |")
    lines.append("")

    lines.append("## Method-Level Next Ideas")
    for idea in method_insights:
        lines.append(f"- {idea}")
    lines.append("")

    lines.append("## Latest Papers (Top 20)")
    for i, paper in enumerate(papers[:20], start=1):
        lines.append(_paper_to_md_line(i, paper))
    lines.append("")

    lines.append("## New Papers Since Last Run (Top 20)")
    if new_papers:
        for i, paper in enumerate(new_papers[:20], start=1):
            lines.append(_paper_to_md_line(i, paper))
    else:
        lines.append("- No new papers detected compared with previous snapshot.")
    lines.append("")
    return "\n".join(lines)


def run_literature_radar(
    workspace: str,
    seed_topic: str,
    engine: str = "openalex",
    max_topics: int = 12,
    per_topic: int = 8,
    max_papers: int = 120,
    year_before: Optional[int] = None,
    year_after: Optional[int] = None,
    recent_years: int = 3,
) -> Dict:
    workspace_input = Path(workspace).expanduser()
    if workspace_input.is_symlink():
        raise ValueError(f"unsafe workspace path (symlink not allowed): {workspace_input}")

    workspace_path = workspace_input.resolve(strict=False)
    if not workspace_path.exists() or not workspace_path.is_dir():
        raise ValueError(f"workspace not found or not a directory: {workspace_path}")
    if workspace_path.is_symlink():
        raise ValueError(f"unsafe workspace path (symlink not allowed): {workspace_path}")

    workspace_markers = ("notes.txt", "workflow_idea.json", "experiment.py", "artifacts", "uploads")
    if not any((workspace_path / marker).exists() for marker in workspace_markers):
        raise ValueError(
            f"unsafe workspace path (not a PaperForge workspace): {workspace_path}"
        )

    radar_root = workspace_path / "artifacts" / "literature_radar"
    history_dir = radar_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    latest_snapshot_path = radar_root / "latest_snapshot.json"
    previous_snapshot = _load_previous_snapshot(latest_snapshot_path)
    previous_keys = {
        _paper_key(item.get("title", ""), _extract_year(item.get("year")))
        for item in previous_snapshot.get("papers", [])
    }

    topics = _expand_topics(seed_topic=seed_topic, max_topics=max_topics)
    if not topics:
        raise ValueError("seed_topic is empty; please provide a keyword for literature radar.")

    dedup: Dict[Tuple[str, int], Dict] = {}

    for query in topics:
        query = _sanitize_text(query, max_chars=MAX_QUERY_CHARS)
        if not query:
            continue
        try:
            results = search_for_papers(query=query, result_limit=per_topic, engine=engine) or []
        except Exception as exc:
            print(f"[radar] query failed: {query} ({exc})")
            continue

        for raw in results:
            if not isinstance(raw, dict):
                continue
            title = _sanitize_text(raw.get("title"), max_chars=MAX_TITLE_CHARS) or "Untitled"
            year = _extract_year(raw.get("year"))
            if (year_before is not None or year_after is not None) and year is None:
                # When explicit year filters are requested, unknown-year records are excluded
                # to keep filter semantics deterministic.
                continue
            if year_before is not None and year is not None and year >= int(year_before):
                continue
            if year_after is not None and year is not None and year <= int(year_after):
                continue
            venue = _sanitize_text(raw.get("venue"), max_chars=MAX_VENUE_CHARS) or "Unknown"
            authors = _sanitize_text(raw.get("authors"), max_chars=MAX_AUTHORS_CHARS) or "Unknown"
            abstract = _sanitize_text(raw.get("abstract"), max_chars=MAX_ABSTRACT_CHARS)
            cites = _extract_citations(raw.get("citationCount"))
            if not _is_likely_finance_paper(
                seed_topic=seed_topic,
                title=title,
                abstract=abstract,
                venue=venue,
            ):
                continue
            key = _paper_key(title, year)

            if key not in dedup:
                dedup[key] = {
                    "title": title,
                    "year": year,
                    "venue": venue,
                    "authors": authors,
                    "abstract": abstract,
                    "citationCount": cites,
                    "method_themes": _detect_method_themes(title=title, abstract=abstract, venue=venue),
                    "matched_queries": [query],
                }
            else:
                matched = dedup[key].setdefault("matched_queries", [])
                if query not in matched and len(matched) < MAX_MATCHED_QUERIES:
                    matched.append(query)
                dedup[key]["citationCount"] = max(cites, int(dedup[key].get("citationCount", 0)))
                if not dedup[key].get("abstract") and abstract:
                    dedup[key]["abstract"] = abstract

    papers = list(dedup.values())
    papers.sort(
        key=lambda x: (
            int(x.get("year") or 0),
            int(x.get("citationCount") or 0),
            len(x.get("matched_queries", [])),
        ),
        reverse=True,
    )
    papers = papers[: max(1, int(max_papers))]

    current_year = datetime.now().year
    recent_year_threshold = max(1900, current_year - max(1, int(recent_years)) + 1)
    new_papers = [
        p
        for p in papers
        if _paper_key(p.get("title", ""), _extract_year(p.get("year")))
        not in previous_keys
    ]
    new_papers.sort(
        key=lambda x: (int(x.get("year") or 0), int(x.get("citationCount") or 0)),
        reverse=True,
    )

    theme_counts = Counter()
    recent_theme_counts = Counter()
    for paper in papers:
        is_recent = int(paper.get("year") or 0) >= recent_year_threshold
        for theme in paper.get("method_themes", []):
            theme_counts[theme] += 1
            if is_recent:
                recent_theme_counts[theme] += 1

    method_insights = _build_method_insights(
        seed_topic=seed_topic,
        theme_counts=theme_counts,
        recent_theme_counts=recent_theme_counts,
        new_paper_count=len(new_papers),
    )

    generated_at = datetime.now().isoformat()
    snapshot_payload = {
        "generated_at": generated_at,
        "seed_topic": seed_topic,
        "engine": engine,
        "expanded_topics": topics,
        "year_before": year_before,
        "year_after": year_after,
        "recent_year_threshold": recent_year_threshold,
        "papers": papers,
        "new_paper_count": len(new_papers),
        "method_insights": method_insights,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_snapshot_path = history_dir / f"snapshot_{timestamp}_{_slugify(seed_topic)}.json"
    _atomic_write_text(
        history_snapshot_path,
        json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
    )
    _atomic_write_text(
        latest_snapshot_path,
        json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
    )

    report_text = _render_markdown_report(
        seed_topic=seed_topic,
        engine=engine,
        expanded_topics=topics,
        papers=papers,
        new_papers=new_papers,
        method_insights=method_insights,
        generated_at=generated_at,
        recent_year_threshold=recent_year_threshold,
    )
    report_path = workspace_path / "literature_radar_report.md"
    _atomic_write_text(report_path, report_text)

    return {
        "generated_at": generated_at,
        "seed_topic": seed_topic,
        "engine": engine,
        "paper_count": len(papers),
        "new_paper_count": len(new_papers),
        "expanded_topics": topics,
        "recent_year_threshold": recent_year_threshold,
        "method_insights": method_insights,
        "report_path": str(report_path),
        "latest_snapshot_path": str(latest_snapshot_path),
        "history_snapshot_path": str(history_snapshot_path),
    }
