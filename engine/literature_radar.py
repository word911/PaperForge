from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from engine.generate_ideas import search_for_papers
from engine.llm import create_client, get_response_from_llm


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
MAX_ABSTRACT_TRANSLATION_INPUT_CHARS = 1800
MAX_ABSTRACT_TRANSLATION_OUTPUT_CHARS = 2200
DEFAULT_REPORT_PAPER_LIMIT = 50
DEFAULT_TRANSLATION_LIMIT = 50
DEFAULT_ABSTRACT_BACKFILL_LIMIT = 50
ABSTRACT_BACKFILL_RESULT_LIMIT = 5
MAX_ABSTRACT_BACKFILL_QUERY_CHARS = 260
MIN_BACKFILL_TITLE_MATCH_SCORE = 0.78
_SEMANTICSCHOLAR_BACKFILL_DISABLED = False
SERPAPI_SCHOLAR_ENDPOINT = "https://serpapi.com/search.json"
SERPAPI_CONNECT_TIMEOUT = 5
SERPAPI_READ_TIMEOUT = 20
BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_CONNECT_TIMEOUT = 5
BRAVE_READ_TIMEOUT = 20


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


def _resolve_optional_limit(value: Optional[int], *, default_value: int) -> int:
    if value is None:
        return max(0, int(default_value))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default_value))


def _title_match_score(
    target_title: str,
    candidate_title: str,
    *,
    target_year: Optional[int],
    candidate_year: Optional[int],
) -> float:
    target_norm = _normalize_title(target_title)
    candidate_norm = _normalize_title(candidate_title)
    if not target_norm or not candidate_norm:
        return 0.0

    if target_norm == candidate_norm:
        score = 1.0
    else:
        score = SequenceMatcher(None, target_norm, candidate_norm).ratio()
        if target_norm in candidate_norm or candidate_norm in target_norm:
            score = max(score, 0.90 if min(len(target_norm), len(candidate_norm)) >= 24 else 0.84)

    if target_year is not None and candidate_year is not None:
        year_gap = abs(int(target_year) - int(candidate_year))
        if year_gap == 0:
            score += 0.03
        elif year_gap >= 5:
            score -= 0.05
    return max(0.0, min(1.0, score))


def _pick_best_abstract_candidate(
    *,
    target_title: str,
    target_year: Optional[int],
    candidates: List[Dict],
) -> str:
    best_score = 0.0
    best_abstract = ""

    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        candidate_title = _sanitize_text(raw.get("title"), max_chars=MAX_TITLE_CHARS)
        candidate_abstract = _sanitize_text(raw.get("abstract"), max_chars=MAX_ABSTRACT_CHARS)
        if not candidate_title or not candidate_abstract:
            continue
        score = _title_match_score(
            target_title=target_title,
            candidate_title=candidate_title,
            target_year=target_year,
            candidate_year=_extract_year(raw.get("year")),
        )
        if score > best_score:
            best_score = score
            best_abstract = candidate_abstract

    if best_score < MIN_BACKFILL_TITLE_MATCH_SCORE:
        return ""
    return best_abstract


def _extract_year_from_text(text: object) -> Optional[int]:
    if text is None:
        return None
    matches = re.findall(r"\b(?:19|20)\d{2}\b", str(text))
    for match in matches:
        year = _extract_year(match)
        if year is not None:
            return year
    return None


def _search_serpapi_scholar_candidates(query: str, result_limit: int) -> List[Dict]:
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        rsp = requests.get(
            SERPAPI_SCHOLAR_ENDPOINT,
            params={
                "engine": "google_scholar",
                "q": query,
                "api_key": api_key,
                "num": max(1, int(result_limit)),
                "hl": "en",
            },
            timeout=(SERPAPI_CONNECT_TIMEOUT, SERPAPI_READ_TIMEOUT),
        )
        rsp.raise_for_status()
        payload = rsp.json() if rsp.content else {}
    except requests.exceptions.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"[radar] serpapi scholar HTTP error (status={status_code}): {exc}")
        return []
    except Exception as exc:
        print(f"[radar] serpapi scholar query failed: {exc}")
        return []

    results = payload.get("organic_results", [])
    if not isinstance(results, list):
        return []

    normalized: List[Dict] = []
    for item in results[: max(1, int(result_limit))]:
        if not isinstance(item, dict):
            continue
        title = _sanitize_text(item.get("title"), max_chars=MAX_TITLE_CHARS)
        snippet = _sanitize_text(item.get("snippet"), max_chars=MAX_ABSTRACT_CHARS)
        publication_info = item.get("publication_info") if isinstance(item.get("publication_info"), dict) else {}
        summary = _sanitize_text(publication_info.get("summary"), max_chars=240)
        if not title or not snippet:
            continue
        normalized.append(
            {
                "title": title,
                "abstract": snippet,
                "year": _extract_year_from_text(summary) or _extract_year_from_text(item.get("year")),
                "venue": "Google Scholar",
                "authors": "",
                "citationCount": 0,
            }
        )
    return normalized


def _search_brave_web_candidates(query: str, result_limit: int) -> List[Dict]:
    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        rsp = requests.get(
            BRAVE_WEB_SEARCH_ENDPOINT,
            params={
                "q": query,
                "count": min(20, max(1, int(result_limit))),
                "extra_snippets": "true",
                "search_lang": "en",
            },
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            timeout=(BRAVE_CONNECT_TIMEOUT, BRAVE_READ_TIMEOUT),
        )
        rsp.raise_for_status()
        payload = rsp.json() if rsp.content else {}
    except requests.exceptions.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"[radar] brave search HTTP error (status={status_code}): {exc}")
        return []
    except Exception as exc:
        print(f"[radar] brave search query failed: {exc}")
        return []

    web_obj = payload.get("web") if isinstance(payload.get("web"), dict) else {}
    results = web_obj.get("results", [])
    if not isinstance(results, list):
        return []

    normalized: List[Dict] = []
    for item in results[: max(1, int(result_limit))]:
        if not isinstance(item, dict):
            continue
        title = _sanitize_text(item.get("title"), max_chars=MAX_TITLE_CHARS)
        description = _sanitize_text(item.get("description"), max_chars=MAX_ABSTRACT_CHARS)
        extra_snippets_raw = item.get("extra_snippets")
        extra_snippets = extra_snippets_raw if isinstance(extra_snippets_raw, list) else []
        extra_joined = " ".join(str(x) for x in extra_snippets if isinstance(x, str))
        extra_text = _sanitize_text(extra_joined, max_chars=MAX_ABSTRACT_CHARS)
        abstract = description or extra_text
        if not title or not abstract:
            continue

        profile = item.get("profile") if isinstance(item.get("profile"), dict) else {}
        venue = _sanitize_text(profile.get("name"), max_chars=MAX_VENUE_CHARS) or "Brave Search"
        year = _extract_year_from_text(item.get("age")) or _extract_year_from_text(abstract)
        normalized.append(
            {
                "title": title,
                "abstract": abstract,
                "year": year,
                "venue": venue,
                "authors": "",
                "citationCount": 0,
            }
        )
    return normalized


def _search_engine_candidates(query: str, *, engine: str, result_limit: int) -> List[Dict]:
    global _SEMANTICSCHOLAR_BACKFILL_DISABLED
    if engine == "semanticscholar" and _SEMANTICSCHOLAR_BACKFILL_DISABLED:
        return []

    if engine == "serpapi_google_scholar":
        return _search_serpapi_scholar_candidates(query=query, result_limit=result_limit)
    if engine == "brave_web_search":
        return _search_brave_web_candidates(query=query, result_limit=result_limit)

    # `search_for_papers` is decorated with unbounded HTTP backoff. For the
    # backfill path we bypass that wrapper to avoid long stalls on rate limits.
    raw_search = getattr(search_for_papers, "__wrapped__", search_for_papers)
    try:
        results = raw_search(query=query, result_limit=result_limit, engine=engine) or []
    except requests.exceptions.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if engine == "semanticscholar" and status_code == 429:
            _SEMANTICSCHOLAR_BACKFILL_DISABLED = True
            print("[radar] semanticscholar backfill disabled for this run due to HTTP 429.")
            return []
        print(f"[radar] abstract backfill HTTP error ({engine}, status={status_code}): {exc}")
        return []
    except Exception as exc:
        print(f"[radar] abstract backfill query failed ({engine}): {exc}")
        return []
    return [item for item in results if isinstance(item, dict)]


def _backfill_abstract_by_title(title: str, year: Optional[int]) -> Tuple[str, str]:
    query = _sanitize_text(title, max_chars=MAX_ABSTRACT_BACKFILL_QUERY_CHARS)
    if not query:
        return "", ""

    query_variants = [f"\"{query}\"", query]
    engines = ["openalex"]
    if os.getenv("S2_API_KEY", "").strip():
        engines.append("semanticscholar")
    if os.getenv("BRAVE_API_KEY", "").strip():
        engines.append("brave_web_search")
    if os.getenv("SERPAPI_API_KEY", "").strip():
        engines.append("serpapi_google_scholar")
    seen_queries = set()

    for engine in engines:
        for variant in query_variants:
            variant_norm = f"{engine}:{_normalize_topic(variant)}"
            if variant_norm in seen_queries:
                continue
            seen_queries.add(variant_norm)

            candidates = _search_engine_candidates(
                variant,
                engine=engine,
                result_limit=ABSTRACT_BACKFILL_RESULT_LIMIT,
            )
            if not candidates:
                continue
            abstract = _pick_best_abstract_candidate(
                target_title=title,
                target_year=year,
                candidates=candidates,
            )
            if abstract:
                return abstract, engine
    return "", ""


def _translate_abstract_to_zh(
    abstract: str,
    *,
    client: Any,
    model: str,
) -> str:
    source = _sanitize_text(abstract, max_chars=MAX_ABSTRACT_TRANSLATION_INPUT_CHARS)
    if not source:
        return ""

    system_message = (
        "You are an expert bilingual quantitative-finance translator. "
        "Translate English research abstracts into clear, faithful, academic Chinese."
    )
    user_message = (
        "请把下面英文论文摘要翻译为中文学术风格。\n"
        "要求：\n"
        "1) 保留方法、数据、结论与限制；\n"
        "2) 不新增原文没有的信息；\n"
        "3) 用中文输出，2-5句；\n"
        "4) 只输出译文，不要解释。\n\n"
        f"Abstract:\n{source}"
    )
    try:
        translated, _ = get_response_from_llm(
            user_message,
            client,
            model,
            system_message=system_message,
            print_debug=False,
            msg_history=None,
            temperature=0.1,
        )
    except Exception as exc:
        print(f"[radar] abstract translation failed: {exc}")
        return ""
    return _sanitize_text(translated, max_chars=MAX_ABSTRACT_TRANSLATION_OUTPUT_CHARS)


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
    detail_paper_limit: int,
    translated_count: int,
    translation_model: Optional[str],
    abstract_backfilled_count: int,
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
    lines.append(f"- Abstract translations generated: {translated_count}")
    lines.append(f"- Abstracts backfilled (missing->found): {abstract_backfilled_count}")
    lines.append(f"- Translation model: `{translation_model or 'N/A'}`")
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

    detail_limit = min(len(papers), max(0, int(detail_paper_limit)))
    lines.append(f"## Paper Abstract Digest (Top {detail_limit})")
    if detail_limit <= 0:
        lines.append("- No paper details requested.")
        lines.append("")
        return "\n".join(lines)

    for i, paper in enumerate(papers[:detail_limit], start=1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "?")
        venue = paper.get("venue", "Unknown")
        cites = paper.get("citationCount", 0)
        themes = ", ".join(paper.get("method_themes", [])) or "N/A"
        abstract = paper.get("abstract", "") or "N/A"
        abstract_zh = paper.get("abstract_zh", "") or "(未生成翻译)"

        lines.append(f"### {i}. {title}")
        lines.append(f"- Year: {year}")
        lines.append(f"- Venue: {venue}")
        lines.append(f"- Citations: {cites}")
        lines.append(f"- Method themes: {themes}")
        lines.append("- Abstract (EN)")
        lines.append(f"> {abstract}")
        lines.append("- 摘要翻译（ZH）")
        lines.append(f"> {abstract_zh}")
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
    translate_abstracts: bool = True,
    translation_model: str = "gpt-5.2",
    translation_limit: Optional[int] = DEFAULT_TRANSLATION_LIMIT,
    detail_paper_limit: Optional[int] = DEFAULT_REPORT_PAPER_LIMIT,
    backfill_missing_abstracts: bool = True,
    abstract_backfill_limit: Optional[int] = DEFAULT_ABSTRACT_BACKFILL_LIMIT,
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
    previous_translation_map = {
        _paper_key(item.get("title", ""), _extract_year(item.get("year"))): _sanitize_text(
            item.get("abstract_zh"),
            max_chars=MAX_ABSTRACT_TRANSLATION_OUTPUT_CHARS,
        )
        for item in previous_snapshot.get("papers", [])
        if _sanitize_text(item.get("abstract_zh"), max_chars=MAX_ABSTRACT_TRANSLATION_OUTPUT_CHARS)
    }
    previous_abstract_map = {
        _paper_key(item.get("title", ""), _extract_year(item.get("year"))): _sanitize_text(
            item.get("abstract"),
            max_chars=MAX_ABSTRACT_CHARS,
        )
        for item in previous_snapshot.get("papers", [])
        if _sanitize_text(item.get("abstract"), max_chars=MAX_ABSTRACT_CHARS)
    }

    topics = _expand_topics(seed_topic=seed_topic, max_topics=max_topics)
    if not topics:
        raise ValueError("seed_topic is empty; please provide a keyword for literature radar.")

    dedup: Dict[Tuple[str, int], Dict] = {}

    def _ingest_results(query: str, results: List[Dict]) -> None:
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

    def _fetch_and_ingest(query: str, *, result_limit: int) -> None:
        cleaned_query = _sanitize_text(query, max_chars=MAX_QUERY_CHARS)
        if not cleaned_query:
            return
        try:
            results = search_for_papers(query=cleaned_query, result_limit=result_limit, engine=engine) or []
        except Exception as exc:
            print(f"[radar] query failed: {cleaned_query} ({exc})")
            return
        _ingest_results(cleaned_query, [item for item in results if isinstance(item, dict)])

    for query in topics:
        _fetch_and_ingest(query, result_limit=per_topic)

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

    effective_translation_model = _sanitize_text(translation_model, max_chars=64) or "gpt-5.2"
    translate_budget = _resolve_optional_limit(
        translation_limit,
        default_value=DEFAULT_TRANSLATION_LIMIT,
    )
    report_detail_limit = _resolve_optional_limit(
        detail_paper_limit,
        default_value=DEFAULT_REPORT_PAPER_LIMIT,
    )
    if report_detail_limit <= 0:
        report_detail_limit = len(papers)
    if translate_budget <= 0:
        translate_budget = report_detail_limit
    abstract_backfill_budget = _resolve_optional_limit(
        abstract_backfill_limit,
        default_value=DEFAULT_ABSTRACT_BACKFILL_LIMIT,
    )
    if abstract_backfill_budget <= 0:
        abstract_backfill_budget = report_detail_limit

    abstract_backfilled_count = 0
    if backfill_missing_abstracts:
        for paper in papers[:report_detail_limit]:
            if _sanitize_text(paper.get("abstract"), max_chars=MAX_ABSTRACT_CHARS):
                continue
            key = _paper_key(paper.get("title", ""), _extract_year(paper.get("year")))
            cached_abstract = previous_abstract_map.get(key)
            if cached_abstract:
                paper["abstract"] = cached_abstract
                paper["abstract_source"] = "snapshot_cache"
                paper["method_themes"] = _detect_method_themes(
                    title=paper.get("title", ""),
                    abstract=cached_abstract,
                    venue=paper.get("venue", ""),
                )
                abstract_backfilled_count += 1
                continue

            if abstract_backfilled_count >= abstract_backfill_budget:
                continue
            filled_abstract, source_engine = _backfill_abstract_by_title(
                title=paper.get("title", ""),
                year=_extract_year(paper.get("year")),
            )
            if not filled_abstract:
                continue

            paper["abstract"] = filled_abstract
            paper["abstract_source"] = source_engine
            paper["method_themes"] = _detect_method_themes(
                title=paper.get("title", ""),
                abstract=filled_abstract,
                venue=paper.get("venue", ""),
            )
            abstract_backfilled_count += 1

    translation_client = None
    translation_model_used: Optional[str] = None
    if translate_abstracts and papers:
        try:
            translation_client, translation_model_used = create_client(effective_translation_model)
        except Exception as exc:
            print(
                f"[radar] translation disabled (model={effective_translation_model} not available): {exc}"
            )
            translation_client = None
            translation_model_used = None

    translated_count = 0
    for paper in papers[:report_detail_limit]:
        key = _paper_key(paper.get("title", ""), _extract_year(paper.get("year")))
        cached_zh = previous_translation_map.get(key)
        if cached_zh:
            paper["abstract_zh"] = cached_zh
            translated_count += 1
            continue

        abstract = paper.get("abstract", "")
        if (
            not translate_abstracts
            or translation_client is None
            or translated_count >= translate_budget
            or not abstract
        ):
            continue
        translated = _translate_abstract_to_zh(
            abstract,
            client=translation_client,
            model=translation_model_used or effective_translation_model,
        )
        if translated:
            paper["abstract_zh"] = translated
            translated_count += 1

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
        "translation_model": translation_model_used or effective_translation_model,
        "translated_abstract_count": translated_count,
        "abstract_backfilled_count": abstract_backfilled_count,
        "detail_paper_limit": report_detail_limit,
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
        detail_paper_limit=report_detail_limit,
        translated_count=translated_count,
        translation_model=translation_model_used or effective_translation_model,
        abstract_backfilled_count=abstract_backfilled_count,
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
        "translated_abstract_count": translated_count,
        "translation_model": translation_model_used or effective_translation_model,
        "abstract_backfilled_count": abstract_backfilled_count,
        "detail_paper_limit": report_detail_limit,
        "report_path": str(report_path),
        "latest_snapshot_path": str(latest_snapshot_path),
        "history_snapshot_path": str(history_snapshot_path),
    }
