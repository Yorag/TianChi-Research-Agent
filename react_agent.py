"""ReAct agent: single-loop research agent with search + fetch tools."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

try:
    from .config import (
        LLM_MAX_TOKENS,
        LLM_TIMEOUT,
        MAX_FETCH_PAGES,
        MAX_ITERATIONS,
        MAX_PAGE_CHARS,
        MAX_RESULTS_PER_QUERY,
        MAX_SEARCH_QUERIES,
        MAX_TOTAL_SECONDS,
        call_llm,
    )
    from .page_fetcher import fetch_page_content
    from .react_prompt import REACT_PROMPT
    from .search_provider import default_provider as search_provider
    from .utils import extract_domain, format_answer, parse_question
except ImportError:
    from config import (
        LLM_MAX_TOKENS,
        LLM_TIMEOUT,
        MAX_FETCH_PAGES,
        MAX_ITERATIONS,
        MAX_PAGE_CHARS,
        MAX_RESULTS_PER_QUERY,
        MAX_SEARCH_QUERIES,
        MAX_TOTAL_SECONDS,
        call_llm,
    )
    from page_fetcher import fetch_page_content
    from react_prompt import REACT_PROMPT
    from search_provider import default_provider as search_provider
    from utils import extract_domain, format_answer, parse_question

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────


@dataclass
class ReActState:
    question: str
    lang: str = "en"
    answer_kind: str = "entity"
    format_hint: str = ""
    format_example: str = ""
    trace: list[dict] = field(default_factory=list)
    searches_used: int = 0
    search_budget: int = MAX_SEARCH_QUERIES
    fetch_budget: int = MAX_FETCH_PAGES
    fetches_used: int = 0
    start_time: float = 0.0
    final_answer: str = ""
    findings: dict[str, str] = field(default_factory=dict)
    seen_urls: set[str] = field(default_factory=set)  # Cross-query URL dedup
    seen_snippets: list[str] = field(default_factory=list)  # Cross-query content dedup


# ── Helpers ──────────────────────────────────────────────────────────


def _time_remaining(state: ReActState) -> float:
    return max(0.0, MAX_TOTAL_SECONDS - (time.time() - state.start_time))


def _parse_react_output(raw: str) -> dict:
    """Parse LLM output into Thought / Action / Action Input / Findings."""
    thought = ""
    action = ""
    action_input = ""
    findings: list[tuple[str, str]] = []
    fabricated = False  # True if LLM produced fake Observation/Result content

    text = raw.strip().replace("\r\n", "\n")

    # Truncate at fabricated content: LLM should never produce Observation/Result,
    # nor should it write multiple Action blocks in one output.
    # Strategy: keep only the FIRST Action block. Truncate at the earliest of:
    #   - fabricated Observation:/Result: after first Action Input
    #   - a second Action: block (LLM continuing to hallucinate multi-step chains)
    _ai_first = re.search(r"Action Input:", text)
    if _ai_first:
        after_first_ai = _ai_first.end()
        # Find fabricated Observation/Result or a second Action block after first Action Input
        _fab = re.search(r"\nObservation:|\nResult:", text[after_first_ai:])
        _second_action = re.search(r"\n\s*Action:\s*\S+", text[after_first_ai:])
        # Also detect Finding: followed by Action: (LLM uses Finding as fake observation)
        _finding_then_action = re.search(r"\nFinding:.*?\nAction:", text[after_first_ai:], re.DOTALL)

        cut_positions = []
        fab_reason = ""
        if _fab:
            cut_positions.append(after_first_ai + _fab.start())
            fab_reason = "fabricated Observation/Result"
        if _second_action:
            pos = after_first_ai + _second_action.start()
            if not cut_positions or pos < min(cut_positions):
                fab_reason = "multiple Action blocks"
            cut_positions.append(pos)
        if _finding_then_action:
            pos = after_first_ai + _finding_then_action.start()
            if not cut_positions or pos < min(cut_positions):
                fab_reason = "Finding-then-Action chain"
            cut_positions.append(pos)

        if cut_positions:
            cut_at = min(cut_positions)
            fabricated_len = len(text) - cut_at
            text = text[:cut_at]
            fabricated = True
            logger.debug(f"  truncated {fabricated_len} chars of fabricated content ({fab_reason})")

    # Parse Finding: lines (e.g. "Finding: 评分系统名称 = Elo等级分")
    for fm in re.finditer(r"Finding:\s*(.+?)\s*=\s*(.+?)(?:\n|$)", text):
        key = fm.group(1).strip()
        val = fm.group(2).strip()
        if key and val:
            findings.append((key, val))

    # Regex extraction
    thought_m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
    action_m = re.search(r"Action:\s*(\S+)", text)
    input_m = re.search(r"Action Input:\s*(.+?)(?=\nThought:|\nFinding:|\nObservation:|\nResult:|\Z)", text, re.DOTALL)

    if thought_m:
        thought = thought_m.group(1).strip()
    if action_m:
        action = action_m.group(1).strip().lower()
    if input_m:
        action_input = input_m.group(1).strip()
        # For search action, take only the first line (LLM sometimes appends reasoning after query)
        if action == "search":
            action_input = action_input.split("\n")[0].strip()

    # Fallback: "Final Answer:" or "答案:"
    if not action:
        m = re.search(r"(?:Final Answer|答案|answer)\s*[:：]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            action = "finish"
            action_input = m.group(1).strip().split("\n")[0]

    # Normalize action names
    _ACTION_NORM = {"search": "search", "fetch": "fetch", "finish": "finish",
                    "final_answer": "finish", "done": "finish"}
    action = _ACTION_NORM.get(action, action)

    # Strip matched surrounding quotes from action_input (only full wrapping pairs)
    if action_input:
        _QUOTE_PAIRS = [('"', '"'), ("'", "'"), ('\u201c', '\u201d'),
                        ('\u300c', '\u300d'), ('\u300e', '\u300f')]
        for q_open, q_close in _QUOTE_PAIRS:
            if (action_input.startswith(q_open) and action_input.endswith(q_close)
                    and len(action_input) > len(q_open) + len(q_close)):
                action_input = action_input[len(q_open):-len(q_close)].strip()
                break

    return {"thought": thought, "action": action, "action_input": action_input,
            "findings": findings, "fabricated": fabricated}




def _snippet_overlap(a: str, b: str) -> float:
    """Compute character-level overlap ratio between two snippets."""
    if not a or not b:
        return 0.0
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / min(len(a_set), len(b_set))


def _format_search_results(
    results: list[dict],
    seen_urls: set[str] | None = None,
    seen_snippets: list[str] | None = None,
) -> str:
    """Format search results as readable text for Observation, preserving search engine ranking."""
    if not results:
        return "(No results found)"

    capped = results[:MAX_RESULTS_PER_QUERY]

    # ── Cross-query dedup ──
    deduped = []
    for r in capped:
        url = r.get("url", "")
        snippet = r.get("snippet", "")

        # URL dedup: skip if exact same URL was seen in a previous query
        if seen_urls is not None and url and url in seen_urls:
            continue

        # Snippet dedup: skip if snippet is highly similar to a previously seen one
        if seen_snippets is not None and snippet:
            if any(_snippet_overlap(snippet, prev) >= 0.9 for prev in seen_snippets):
                continue

        deduped.append(r)

        # Record for future dedup
        if seen_urls is not None and url:
            seen_urls.add(url)
        if seen_snippets is not None and snippet:
            seen_snippets.append(snippet)

    if not deduped:
        return "(All results duplicated from previous searches)"

    lines = []
    for i, r in enumerate(deduped):
        # Source type tags (useful context for LLM)
        source_tag = ""
        source_type = r.get("source_type", "organic")
        if source_type == "knowledgeGraph":
            source_tag = "[Knowledge Graph] "
        elif source_type == "answerBox":
            source_tag = "[Answer Box] "
        domain = r.get("domain", "") or extract_domain(r.get("url", ""))
        if "wikipedia" in domain or "baike.baidu" in domain:
            source_tag += "[Encyclopedia] "

        parts = [f"[{i+1}] {source_tag}{r.get('title', '')}"]
        snippet = r.get("snippet", "")
        if snippet:
            parts.append(f"  {snippet}")
        url = r.get("url", "")
        if url:
            parts.append(f"  URL: {url}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


_RECENT_FULL_STEPS = 5  # Keep full Result for the most recent N steps

def _build_trace_text(state: ReActState) -> str:
    """Build the trace text to inject into the prompt.

    Recent steps (last _RECENT_FULL_STEPS) keep full Result.
    Older steps keep Thought/Action/Action Input intact but replace
    Result with a short note — key facts are already captured in Findings.

    NOTE: We use 'Result:' instead of 'Observation:' in the prompt to prevent
    LLM from learning the Thought→Action→Action Input→Observation pattern and
    fabricating Observation content (hallucination).
    """
    if not state.trace:
        return "(No previous steps. Start your research.)"

    total = len(state.trace)
    cutoff = total - _RECENT_FULL_STEPS  # index where full display starts

    lines = []
    for i, entry in enumerate(state.trace):
        lines.append(f"Thought: {entry.get('thought', '')}")
        lines.append(f"Action: {entry.get('action', '')}")
        lines.append(f"Action Input: {entry.get('action_input', '')}")
        if i < cutoff:
            # Old step: omit verbose Result
            obs = entry.get('observation', '')
            if len(obs) > 400:
                lines.append("Result: (omitted — see Findings for key facts)")
            else:
                lines.append(f"Result: {obs}")
        else:
            lines.append(f"Result: {entry.get('observation', '')}")
        lines.append("")

    return "\n".join(lines)


def _build_prompt(state: ReActState) -> str:
    """Assemble the complete prompt with trace and budget info."""
    # Build findings summary
    findings_text = ""
    if state.findings:
        lines = [f"- {k}: {v}" for k, v in state.findings.items()]
        findings_text = "\n".join(lines)
        logger.debug(f"  findings snapshot: { {k: v[:40] for k, v in state.findings.items()} }")

    # Build format cue from format_example
    format_cue = ""
    if state.format_example:
        format_cue = (
            f"\n\n## Answer Format Constraint\n"
            f"The question specifies a format example: \"{state.format_example}\". "
            f"Your final answer MUST follow this exact format style. "
            f"Match the naming convention (e.g., use \"Limited\" not \"Ltd.\", "
            f"use \"Group\" if the example includes it). Study the example carefully."
        )

    return REACT_PROMPT.format(
        question=state.question,
        lang=state.lang,
        format_cue=format_cue,
        search_remaining=state.search_budget - state.searches_used,
        search_budget=state.search_budget,
        searches_used=state.searches_used,
        fetch_remaining=state.fetch_budget - state.fetches_used,
        time_remaining=_time_remaining(state),
        trace=_build_trace_text(state),
        findings=findings_text,
    )



_GIVE_UP_PATTERNS = re.compile(
    r"unable to determine|cannot be determined|无法确定|无法判断|cannot determine|"
    r"not enough information|insufficient|no (?:reliable |credible |verifiable )?(?:answer|result|information|source)|"
    r"cannot (?:find|identify|confirm|provide)|could not (?:find|determine|identify)|"
    r"无法找到|无法回答|信息不足|not determinable|indeterminate|unknown",
    re.IGNORECASE,
)


def _is_give_up_answer(answer: str) -> bool:
    """Check if an answer is a give-up / refusal rather than a real answer."""
    if not answer:
        return True
    # Very long answers (>200 chars) are likely explanations, not concise answers
    if len(answer) > 200:
        return True
    return bool(_GIVE_UP_PATTERNS.search(answer))



def _extract_answer_from_findings(state: ReActState) -> str:
    """Try to extract a concise answer from confirmed findings."""
    if not state.findings:
        return ""

    # Priority: findings whose key mentions answer-related terms
    answer_keys = []
    other_keys = []
    for k, v in state.findings.items():
        k_lower = k.lower()
        if any(w in k_lower for w in ("答案", "answer", "最终", "final", "结果", "result")):
            answer_keys.append((k, v))
        else:
            other_keys.append((k, v))

    # Check answer-keyed findings first, then fall back to last finding
    candidates = answer_keys if answer_keys else other_keys
    if candidates:
        # Take the last one (most recent = most refined)
        _, val = candidates[-1]
        val = val.strip()
        if val and len(val) < 80:
            return val
    return ""


def _llm_fallback_extraction(state: ReActState) -> str:
    """Use LLM to summarize trace into an answer (enhanced with findings)."""
    if not state.trace:
        return ""
    try:
        trace_summary = ""
        for entry in state.trace[-8:]:
            t = entry.get("thought", "")
            if t:
                trace_summary += f"Thought: {t}\n"
            if entry.get("action") == "search":
                obs = entry.get("observation", "")
                trace_summary += f"Search '{entry.get('action_input', '')}': {obs}\n\n"

        # Inject findings as confirmed facts
        findings_text = ""
        if state.findings:
            lines = [f"- {k}: {v}" for k, v in state.findings.items()]
            findings_text = "Confirmed facts:\n" + "\n".join(lines) + "\n\n"

        # Add answer_kind hint
        kind_hint = ""
        if state.answer_kind == "number":
            kind_hint = " The answer should be a number."
        elif state.answer_kind == "date":
            kind_hint = " The answer should be a date."
        elif state.format_hint:
            kind_hint = f" Format hint: {state.format_hint}"
        if state.format_example:
            kind_hint += f" The answer format must follow the example: \"{state.format_example}\""

        prompt = (
            f"Based on the research below, what is the concise answer to: {state.question}\n\n"
            f"{findings_text}"
            f"{trace_summary}\n"
            f"Output ONLY the answer (a name, number, or short phrase). No explanation.{kind_hint}\n"
            f"IMPORTANT: Only use names/entities that appeared verbatim in search results or snippets. "
            f"Do NOT transliterate foreign names yourself — copy them exactly as they appear in the sources."
        )
        answer = call_llm(prompt, temperature=0.0, timeout=30).strip()
        if answer and len(answer) < 80:
            logger.debug(f"LLM fallback extracted: {answer}")
            return answer
    except Exception as e:
        logger.debug(f"LLM fallback extraction failed: {e}")
    return ""


def _answer_kind_bonus(candidate: str, state: ReActState) -> float:
    """Return a small bonus/penalty based on answer_kind match."""
    if state.answer_kind == "number":
        if re.search(r"\d", candidate):
            return 0.1
        return -0.1
    if state.answer_kind == "date":
        if re.search(r"\d", candidate):
            return 0.1
        return -0.1
    return 0.0


def _regex_thought_extraction(state: ReActState) -> str:
    """Last-resort: scan Thought text for answer-like patterns. Only used when LLM is unavailable."""
    if not state.trace:
        return ""

    # Scan thoughts from latest to earliest
    for entry in reversed(state.trace):
        thought = entry.get("thought", "")
        if not thought:
            continue

        # Pattern 1: "答案是/应该是 X" or "the answer is X"
        m = re.search(
            r'(?:答案[是为应]|answer\s+(?:is|should be))\s*[:：]?\s*(.+?)(?:[。\.\n]|$)',
            thought, re.IGNORECASE
        )
        if m:
            val = m.group(1).strip().strip('"\'""''')
            if val and len(val) < 80:
                return val

        # Pattern 2: Explicit "**answer**" in markdown bold
        m = re.search(r'\*\*(.{2,60})\*\*', thought)
        if m:
            candidate = m.group(1).strip()
            # Filter out generic terms that are not answers
            if not re.match(r'(?:thought|action|observation|note|summary)', candidate, re.IGNORECASE):
                return candidate

    return ""


def _extract_best_answer_from_trace(
    state: ReActState,
    answer_candidates: list[tuple[str, int, str]] | None = None,
) -> str:
    """If the loop ended without finish, extract the best answer using multi-source scoring."""
    logger.info("extracting answer from trace (no finish action)")

    if answer_candidates is None:
        answer_candidates = []

    scored: list[tuple[str, float, str]] = []  # (answer, confidence, source)

    # Source 1: finish proposal history (highest confidence)
    if answer_candidates:
        for ans, iteration, source in answer_candidates:
            # Later proposals get slightly higher confidence
            base = 0.90 + min(iteration * 0.002, 0.05)
            score = base + _answer_kind_bonus(ans, state)
            scored.append((ans, score, f"finish_proposal@iter{iteration}"))
            logger.debug(f"  candidate from {source}@iter{iteration}: '{ans}' score={score:.2f}")

    # Source 2: findings extraction
    findings_answer = _extract_answer_from_findings(state)
    if findings_answer:
        score = 0.75 + _answer_kind_bonus(findings_answer, state)
        scored.append((findings_answer, score, "findings"))
        logger.debug(f"  candidate from findings: '{findings_answer}' score={score:.2f}")

    # Source 3: LLM fallback (always try when no finish proposal)
    best_so_far = max((s for _, s, _ in scored), default=0.0)
    if best_so_far < 0.85:
        llm_answer = _llm_fallback_extraction(state)
        if llm_answer:
            score = 0.70 + _answer_kind_bonus(llm_answer, state)
            scored.append((llm_answer, score, "llm_fallback"))
            logger.debug(f"  candidate from LLM fallback: '{llm_answer}' score={score:.2f}")

    # Source 4: Regex fallback from Thought text (last resort, no LLM needed)
    if not scored:
        regex_answer = _regex_thought_extraction(state)
        if regex_answer:
            score = 0.50 + _answer_kind_bonus(regex_answer, state)
            scored.append((regex_answer, score, "regex_thought"))
            logger.debug(f"  candidate from regex_thought: '{regex_answer}' score={score:.2f}")

    if not scored:
        logger.debug("  no candidates found")
        return ""

    # Select the highest-confidence candidate
    scored.sort(key=lambda x: x[1], reverse=True)
    best_answer, best_score, best_source = scored[0]
    logger.info(f"  selected: '{best_answer}' (score={best_score:.2f}, source={best_source})")
    return best_answer


# ── Main loop ────────────────────────────────────────────────────────


async def run_react_agent(question: str) -> str:
    """Entry point: run the ReAct loop and return a concise answer."""
    logger.info(f"=== ReAct start: {question[:80]} ===")

    # 1. Parse question
    parsed = parse_question(question)
    state = ReActState(
        question=question,
        lang=parsed["lang"],
        answer_kind=parsed["answer_kind"],
        format_hint=parsed.get("format_hint", ""),
        format_example=parsed.get("format_example", ""),
        search_budget=MAX_SEARCH_QUERIES,
        fetch_budget=MAX_FETCH_PAGES,
        start_time=time.time(),
    )

    consecutive_errors = 0
    verification_count = 0  # Track how many verification rounds have been done
    verification_searches = 0  # Number of real searches done after falsification prompt
    verified_answer = ""  # The answer currently being verified
    answer_switches = 0  # How many times the answer has been switched during verification
    consecutive_blocks = 0  # Count consecutive blocks to the SAME answer (detect dead loop)
    last_blocked_answer = ""  # The answer being repeatedly blocked
    past_queries: list[str] = []  # Normalized past search queries for exact-duplicate detection
    answer_candidates: list[tuple[str, int, str]] = []  # (answer, iteration, source)

    # 2. ReAct loop
    for iteration in range(MAX_ITERATIONS):
        step_start = time.time()
        # Time guard
        remaining = _time_remaining(state)
        if remaining < 15:
            logger.info(f"iter {iteration}: time nearly up ({remaining:.0f}s), forcing finish")
            break

        # Budget guard: nothing left to do
        if (state.searches_used >= state.search_budget
                and state.fetches_used >= state.fetch_budget):
            logger.info(f"iter {iteration}: all budgets exhausted")
            break

        # Build prompt and call LLM
        prompt = _build_prompt(state)
        logger.debug(f"iter {iteration}: prompt length = {len(prompt)} chars")
        try:
            # Dynamic timeout: use at most 40% of remaining time, clamped to [30, LLM_TIMEOUT]
            remaining_for_timeout = _time_remaining(state)
            llm_timeout = min(LLM_TIMEOUT, max(30, int(remaining_for_timeout * 0.4)))
            raw_output = call_llm(prompt, temperature=0.0, timeout=llm_timeout)
            logger.debug(f"iter {iteration}: LLM output length = {len(raw_output)} chars")
            logger.debug(f"iter {iteration}: LLM output START >>>>\n{raw_output[:2000]}\n<<<< END (truncated to 2000)")
            consecutive_errors = 0
        except Exception as e:
            logger.warning(f"iter {iteration}: LLM error: {e}")
            consecutive_errors += 1
            if consecutive_errors >= 3:
                break
            continue

        # Parse output
        parsed_output = _parse_react_output(raw_output)
        thought = parsed_output["thought"]
        action = parsed_output["action"]
        action_input = parsed_output["action_input"]
        output_fabricated = parsed_output.get("fabricated", False)

        # Record findings from this step
        for fkey, fval in parsed_output.get("findings", []):
            state.findings[fkey] = fval
            logger.debug(f"  finding: {fkey} = {fval}")

        logger.info(f"iter {iteration}: action={action}, input={action_input[:80] if action_input else ''}")

        # Safety net: if parser chose a non-finish action but the raw output
        # also contains a finish proposal, capture it as a candidate for timeout
        # recovery.  SKIP when output_fabricated — the finish in fabricated
        # content is part of a hallucinated chain and the answer was never
        # grounded in real search results.
        if action and action != "finish" and not output_fabricated:
            finish_m = re.search(
                r"Action:\s*finish\s*\n\s*Action Input:\s*(.+?)(?=\n|$)",
                raw_output, re.IGNORECASE,
            )
            if finish_m:
                hidden_answer = finish_m.group(1).strip().strip("\"'""「」")
                if (hidden_answer and len(hidden_answer) < 200
                        and not _is_give_up_answer(hidden_answer)):
                    answer_candidates.append((hidden_answer, iteration, "hidden_finish"))
                    logger.debug(f"  hidden finish proposal: {hidden_answer}")

        # Unparseable output
        if not action:
            # Likely truncated by max_tokens — long Thought but no Action
            # LLM_MAX_TOKENS tokens ≈ LLM_MAX_TOKENS*3 chars for mixed CJK/EN text
            truncation_threshold = LLM_MAX_TOKENS * 3
            if thought and len(raw_output) > truncation_threshold:
                logger.debug(f"iter {iteration}: output truncated (len={len(raw_output)}), saving Thought and retrying")
                state.trace.append({
                    "thought": thought,
                    "action": "truncated",
                    "action_input": "",
                    "observation": "OUTPUT TRUNCATED: Your Thought was too long and got cut off before Action. "
                                   "Keep Thought to 2-5 sentences. Now output ONLY Action and Action Input.",
                })
                continue
            logger.debug(f"iter {iteration}: no action parsed")
            consecutive_errors += 1
            if consecutive_errors >= 3:
                break
            state.trace.append({
                "thought": thought or raw_output[:200],
                "action": "error",
                "action_input": "",
                "observation": "FORMAT ERROR: Use exactly: Thought: / Action: / Action Input:",
            })
            continue

        consecutive_errors = 0
        observation = ""

        # ── Execute action ──

        if action == "search":
            query = action_input.strip()
            # Parse optional language prefix: [en] or [zh]
            search_hl = None
            lang_prefix_m = re.match(r"^\[(en|zh)\]\s*", query, re.IGNORECASE)
            if lang_prefix_m:
                prefix_lang = lang_prefix_m.group(1).lower()
                search_hl = "en" if prefix_lang == "en" else "zh-cn"
                query = query[lang_prefix_m.end():].strip()

            # If query too long for Serper (>200 chars), ask LLM to condense with context
            if len(query) > 200:
                logger.debug(f"  query too long ({len(query)} chars), asking LLM to condense")
                try:
                    condensed = call_llm(
                        f"原始问题：{state.question}\n\n"
                        f"当前推理意图：{thought}\n\n"
                        f"过长的搜索词：{query}\n\n"
                        f"请基于以上上下文，将搜索词压缩为≤5个关键词（≤80字符）的简短搜索词，"
                        f"保留核心语义和关键实体。只输出搜索词，不要解释。",
                        temperature=0.0,
                    ).strip().strip("\"'")
                    if condensed and len(condensed) < 200:
                        logger.debug(f"  condensed query: {condensed}")
                        query = condensed
                except Exception:
                    pass  # Use original query, Serper may still handle it

            if not query:
                observation = "ERROR: empty search query."
            elif state.searches_used >= state.search_budget:
                observation = "BUDGET EXHAUSTED: No searches left. Use Action: finish now."
            else:
                # Exact-duplicate detection (normalized: lowercase + collapsed whitespace)
                query_norm = " ".join(query.lower().split())
                if query_norm in past_queries:
                    logger.info(f"  search BLOCKED (exact duplicate): {query}")
                    observation = (
                        f"DUPLICATE SEARCH BLOCKED: You already searched '{query}' — repeating it wastes budget.\n"
                        f"You MUST change your approach. Try one of:\n"
                        f"- Switch language (Chinese↔English)\n"
                        f"- Search a DIFFERENT constraint from the question\n"
                        f"- Fetch a URL from earlier results for more details\n"
                        f"- Use fewer/different keywords (2-3 core terms only)\n"
                        f"- If you have enough evidence, use Action: finish"
                    )
                    state.trace.append({
                        "thought": thought, "action": action,
                        "action_input": action_input, "observation": observation,
                    })
                    continue
                past_queries.append(query_norm)

                try:
                    kwargs = {"num": MAX_RESULTS_PER_QUERY}
                    if search_hl:
                        kwargs["hl"] = search_hl
                    results = await search_provider.search(query, **kwargs)
                    state.searches_used += 1
                    # Count real searches performed after falsification prompt
                    # (only if results were returned — empty results don't count)
                    if verification_count >= 1 and len(results) > 0:
                        verification_searches += 1
                    observation = _format_search_results(results, seen_urls=state.seen_urls, seen_snippets=state.seen_snippets)
                    logger.info(f"  search: {len(results)} results, used {state.searches_used}/{state.search_budget}")
                    # Log individual result titles+URLs+snippets for diagnostics
                    for _ri, _r in enumerate(results[:MAX_RESULTS_PER_QUERY]):
                        _snippet = _r.get('snippet', '')
                        _snippet_preview = _snippet[:120] + '...' if len(_snippet) > 120 else _snippet
                        logger.debug(f"    [{_ri+1}] {_r.get('title', '')[:60]} | {_r.get('url', '')}")
                        if _snippet_preview:
                            logger.debug(f"        {_snippet_preview}")

                    # Hint to use fetch when encyclopedia URLs are present
                    if results and state.fetches_used < state.fetch_budget:
                        has_encyclopedia = any(
                            "wikipedia.org" in r.get("url", "") or "baike.baidu" in r.get("url", "")
                            for r in results
                        )
                        if has_encyclopedia:
                            observation += (
                                "\n\n(TIP: Encyclopedia URL found above. "
                                "Consider fetch() for more details if snippets are insufficient.)"
                            )
                except Exception as e:
                    observation = f"Search error: {e}"

        elif action == "fetch":
            url = action_input.strip()
            if not url or not url.startswith("http"):
                observation = "ERROR: Invalid URL. Must start with http."
            elif state.fetches_used >= state.fetch_budget:
                observation = "FETCH BUDGET EXHAUSTED. Use search or finish."
            elif _time_remaining(state) < 30:
                observation = "Not enough time for fetch. Use finish."
            else:
                try:
                    content = await fetch_page_content(url)
                    state.fetches_used += 1
                    observation = content[:MAX_PAGE_CHARS] if content else "Fetch returned empty. Try another URL."
                    logger.info(f"  fetch: {len(content) if content else 0} chars")
                    if content:
                        logger.debug(f"  fetch preview: {content[:200]}...")
                except Exception as e:
                    observation = f"Fetch error: {e}"

        elif action == "finish":
            answer_candidate = action_input.strip()
            budget_left = state.search_budget - state.searches_used

            # Detect give-up answers (e.g., "unable to determine", "无法确定")
            is_give_up = _is_give_up_answer(answer_candidate)

            # Block give-up when budget is still available
            if is_give_up and budget_left > 5:
                logger.info(f"  [BLOCKED] finish blocked (give-up with {budget_left} searches left): {answer_candidate}")
                observation = (
                    f"REJECTED: You still have {budget_left} searches remaining. Do NOT give up.\n"
                    f"Try a COMPLETELY DIFFERENT search strategy:\n"
                    f"- Break the question into independent sub-constraints and search for the MOST UNIQUE one alone (e.g., a rare name, a specific date range, a niche topic).\n"
                    f"- Try searching for just 2-3 core keywords instead of many.\n"
                    f"- Switch language (Chinese↔English).\n"
                    f"- Search for background entities first (e.g., 'universities founded 1985' or 'dating apps PhD thesis 2022').\n"
                    f"- Use fetch() on a promising URL from earlier results.\n"
                    f"You MUST keep trying until budget is nearly exhausted."
                )
                state.trace.append({
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": observation,
                })
                continue

            # Save candidate for timeout recovery (skip give-up answers)
            if answer_candidate and not is_give_up:
                answer_candidates.append((answer_candidate, iteration, "finish_proposal"))

            # ── Verification state machine ──
            # Phase 0: first finish → require falsification search
            # Phase 1: searched but not yet confirmed → require final confirmation
            # Phase 2+: confirmed → accept
            #
            # If LLM switches to a DIFFERENT answer mid-verification, reset the
            # state machine so the new answer also gets properly verified.
            needs_verification = (
                answer_candidate and not is_give_up and budget_left > 3
            )
            if (needs_verification and verified_answer
                    and answer_candidate.lower() != verified_answer.lower()):
                answer_switches += 1
                if answer_switches >= 3:
                    # Track consecutive blocks to the SAME answer to detect dead loops
                    if answer_candidate.lower() == last_blocked_answer.lower():
                        consecutive_blocks += 1
                    else:
                        consecutive_blocks = 1
                        last_blocked_answer = answer_candidate

                    # Dead loop escape: if LLM insists on the same alternative 3+ times
                    # consecutively, it likely has strong evidence — accept the switch
                    if consecutive_blocks >= 3:
                        logger.info(f"  [VERIFY] dead loop detected: LLM insisted on '{answer_candidate}' "
                                    f"{consecutive_blocks} times, accepting switch")
                        verification_count = 0
                        verification_searches = 0
                        verified_answer = ""
                        consecutive_blocks = 0
                        last_blocked_answer = ""
                        # Fall through to normal verification flow below
                    else:
                        # Block repeated answer switching — record as candidate but keep original
                        logger.info(f"  [VERIFY] answer switch BLOCKED (switch #{answer_switches}): "
                                    f"'{verified_answer}' → '{answer_candidate}' (kept original)")
                        answer_candidates.append((answer_candidate, iteration, "blocked_switch"))
                        observation = (
                            f"ANSWER SWITCH BLOCKED: You tried to change from '{verified_answer}' "
                            f"to '{answer_candidate}' during falsification. This is NOT allowed "
                            f"unless search results EXPLICITLY CONTRADICT '{verified_answer}' "
                            f"(e.g., a source says '{verified_answer}' is wrong/incorrect/not X). "
                            f"Soft differences (fame, style, ordering) do NOT justify switching. "
                            f"Continue verifying '{verified_answer}'. "
                            f"If you found decisive negative evidence, state it explicitly and try finish again."
                        )
                        state.trace.append({
                            "thought": thought, "action": action,
                            "action_input": action_input, "observation": observation,
                        })
                        continue
                else:
                    # First switch allowed — reset to Phase 0 for the new answer
                    logger.info(f"  [VERIFY] answer changed '{verified_answer}' → '{answer_candidate}' (switch #{answer_switches}), resetting verification")
                    verification_count = 0
                    verification_searches = 0
                    verified_answer = ""

            if needs_verification and verification_count == 0:
                # Phase 0 → 1: demand a falsification search
                verification_count = 1
                verification_searches = 0
                verified_answer = answer_candidate
                logger.info(f"  [VERIFY] finish deferred for falsification: {answer_candidate}")
                observation = (
                    f"FALSIFICATION CHECK: You proposed '{answer_candidate}'. "
                    f"BEFORE confirming:\n"
                    f"1) List EVERY constraint from the original question.\n"
                    f"2) Check if ANY constraint is NOT confirmed by search evidence. "
                    f"If so, your answer is likely WRONG — do NOT rationalize the gap.\n"
                    f"3) Scan ALL search result snippets from your trace: if any snippet mentions "
                    f"an entity name you have NOT investigated, search for it to confirm or rule it out "
                    f"— but this does NOT automatically mean your current answer is wrong.\n"
                    f"4) Search for an ALTERNATIVE answer using the question's MOST DISCRIMINATING constraint "
                    f"WITHOUT your current candidate name.\n"
                    f"5) **Switching standard**: You may ONLY switch to a different answer if the new candidate "
                    f"satisfies a SPECIFIC CONSTRAINT from the question that '{answer_candidate}' FAILS. "
                    f"Soft signals (name ordering in lists, fame, symbolic associations) are NOT grounds for switching. "
                    f"If both candidates satisfy the same set of constraints, KEEP your original answer '{answer_candidate}'."
                )
            elif needs_verification and verification_count == 1 and verification_searches < 2:
                # Still in Phase 1 but no search yet → reject
                logger.info(f"  [VERIFY] finish rejected — no search done since falsification: {answer_candidate}")
                observation = (
                    f"REJECTED: You must SEARCH for an alternative before confirming '{answer_candidate}'. "
                    f"Try a search that could find a DIFFERENT answer (e.g., rephrase the question, "
                    f"search for specific constraints excluding '{answer_candidate}'). "
                    f"Do NOT repeat finish without searching first."
                )
                state.trace.append({
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": observation,
                })
                continue
            elif needs_verification and verification_count == 1 and verification_searches >= 2:
                # Phase 1 → 2: searched, now do final check
                verification_count = 2
                logger.info(f"  [VERIFY] finish deferred for final check: {answer_candidate}")
                observation = (
                    f"FINAL CHECK: You still propose '{answer_candidate}'. "
                    f"Review the search results from your falsification search. "
                    f"Did you find any alternative that satisfies a SPECIFIC CONSTRAINT "
                    f"that '{answer_candidate}' FAILS? If yes, switch to that alternative. "
                    f"If all alternatives satisfy the SAME constraints as '{answer_candidate}' "
                    f"(i.e., no constraint advantage), KEEP '{answer_candidate}' — do NOT switch "
                    f"based on fame, list ordering, or symbolic associations. "
                    f"If no better alternative exists, confirm with finish."
                )
            else:
                # Phase 2+ or no budget: accept
                # But never accept give-up answers as final — let fallback extract a real one
                if answer_candidate and not is_give_up:
                    state.final_answer = answer_candidate
                    logger.info(f"  finish: {state.final_answer}")
                elif is_give_up:
                    logger.info(f"  [BLOCKED] give-up accepted as loop exit but not as answer: {answer_candidate}")
                break

        else:
            observation = f"Unknown action '{action}'. Use: search, fetch, or finish."

        step_elapsed = time.time() - step_start
        total_elapsed = time.time() - state.start_time
        logger.info(f"iter {iteration}: step={step_elapsed:.1f}s | total={total_elapsed:.1f}s | remaining={_time_remaining(state):.0f}s")

        # Warn LLM when it fabricated Observation/Result content
        if output_fabricated and action in ("search", "fetch"):
            observation += (
                "\n\n⚠ WARNING: Your previous output contained FABRICATED Observation/Result "
                "content that was DISCARDED. You must NEVER write Observation or Result — "
                "those are provided by the system. Only output: Thought → Action → Action Input. "
                "The result above is the REAL search/fetch output. Base your reasoning ONLY on it."
            )
            logger.info(f"iter {iteration}: fabricated content warning injected")

        state.trace.append({
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
        })

    # 3. Fallback: extract answer from candidates, findings, or trace
    if not state.final_answer:
        state.final_answer = _extract_best_answer_from_trace(state, answer_candidates)

    # 4. Format answer (clean LLM output artifacts only; no normalization)
    answer = format_answer(state.final_answer, state.answer_kind)

    logger.info(f"=== ReAct done: '{answer}' (searches={state.searches_used}, steps={len(state.trace)}) ===")
    return answer
