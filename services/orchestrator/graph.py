import json
import operator
import re
from difflib import SequenceMatcher
from typing import Annotated, TypedDict

from langfuse.openai import OpenAI
from langgraph.constants import Send
from langgraph.graph import END, StateGraph


client = OpenAI()


# ---------------------------------------------------------------------
# Review configuration
# ---------------------------------------------------------------------

MAX_FINDINGS = 15
MAX_LOW_SEVERITY_FINDINGS = 3

SEVERITY_PRIORITY = {
    "critical": 0,
    "error": 1,
    "high": 1,
    "warning": 2,
    "medium": 2,
    "low": 3,
    "info": 4,
    "style": 4,
}


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

COMMON_RULES = """
You are reviewing ONLY the code contained in the supplied git diff.

Important rules:

1. Only report issues that are directly supported by visible code.
2. Do not invent variables, functions, dependencies, or behavior.
3. Do not report generic best practices unless the visible code
   actually violates them.
4. Prefer no finding over a speculative or low-confidence finding.
5. Do not report trivial formatter-only issues unless readability is
   materially harmed.
6. Every finding must refer to a real file and relevant line visible
   in the diff.
7. Focus on problems that would genuinely help an engineer improve
   production code.
8. Do not repeat the same underlying problem multiple times.

Return ONLY a JSON array.

Every item must have exactly these keys:

{
  "file": "path/to/file.py",
  "line": 123,
  "severity": "critical|high|medium|low|info",
  "title": "Short issue title",
  "message": "Explain the problem, impact, and concrete recommendation",
  "confidence": 0.0
}

confidence must be a number between 0.0 and 1.0.
"""


PROMPTS = {
    "static_analysis": f"""
{COMMON_RULES}

You are a senior static-analysis and code-quality reviewer.

Review for:
- actual bugs
- excessive cyclomatic complexity
- dead or unreachable code
- duplicated logic
- unsafe type assumptions
- incorrect resource management
- incorrect async/sync usage
- meaningful naming problems
- maintainability problems
- exception-handling problems

Do NOT behave like a formatter.
Do not complain about harmless spacing, modern Python list[] syntax,
or stylistic preferences that do not affect maintainability.
""",

    "security": f"""
{COMMON_RULES}

You are a senior application security reviewer.

Review for:
- SQL/command/template injection
- authentication weaknesses
- authorization and IDOR
- secrets and credential handling
- insecure defaults
- sensitive data exposure
- unsafe logging
- cryptographic mistakes
- timing-sensitive comparisons
- webhook signature verification
- webhook replay attacks
- SSRF
- path traversal
- unsafe deserialization
- missing validation at trust boundaries
- privilege escalation
- insecure administrative endpoints
- OWASP-style vulnerabilities

For HMAC or secret comparisons, specifically check whether
constant-time comparison is required.

Do not report hypothetical attacks unless the visible code supports
the finding.
""",

    "architecture": f"""
{COMMON_RULES}

You are a senior software architecture and reliability reviewer.

Review for:
- poor separation of concerns
- incorrect service boundaries
- transaction boundaries
- partial failure handling
- retries and retry safety
- idempotency
- concurrency and race conditions
- distributed-system consistency
- durable vs non-durable background work
- fire-and-forget async work
- timeouts
- retry backoff
- circuit breaking
- external dependency failures
- resource lifecycle
- database consistency
- compensation/rollback behavior
- invalid state transitions
- excessive coupling
- missing error handling

Prioritize production-impacting architectural problems rather than
minor design preferences.
""",
}


# ---------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------

class GraphState(TypedDict):
    diff: str
    patterns: list[str]
    findings: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------

def parse_json_response(raw: str) -> list[dict]:
    raw = raw.strip()

    match = re.search(
        r"```(?:json)?\s*([\s\S]*?)```",
        raw,
    )

    if match:
        raw = match.group(1).strip()

    try:
        parsed = json.loads(raw)

        if not isinstance(parsed, list):
            return []

        return [
            item
            for item in parsed
            if isinstance(item, dict)
        ]

    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_severity(value: str) -> str:
    value = str(value or "info").lower().strip()

    mapping = {
        "error": "high",
        "warning": "medium",
        "style": "low",
    }

    value = mapping.get(value, value)

    if value not in {
        "critical",
        "high",
        "medium",
        "low",
        "info",
    }:
        return "info"

    return value


def normalize_finding(
    finding: dict,
    agent_name: str,
) -> dict | None:

    file_path = finding.get("file")
    message = str(finding.get("message") or "").strip()

    if not file_path or not message:
        return None

    try:
        line = int(finding.get("line") or 0)
    except (TypeError, ValueError):
        line = 0

    try:
        confidence = float(
            finding.get("confidence", 0.75)
        )
    except (TypeError, ValueError):
        confidence = 0.75

    confidence = max(
        0.0,
        min(confidence, 1.0),
    )

    title = str(
        finding.get("title")
        or message[:80]
    ).strip()

    return {
        "file": str(file_path),
        "line": line,
        "severity": normalize_severity(
            finding.get("severity")
        ),
        "title": title,
        "message": message,
        "confidence": confidence,
        "agent": agent_name,
    }


# ---------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------

def passes_confidence_threshold(
    finding: dict,
) -> bool:

    thresholds = {
        "critical": 0.65,
        "high": 0.70,
        "medium": 0.78,
        "low": 0.88,
        "info": 0.93,
    }

    severity = finding["severity"]

    return (
        finding["confidence"]
        >= thresholds[severity]
    )


# ---------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------

def normalize_text(value: str) -> str:
    value = value.lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    return " ".join(value.split())


def text_similarity(
    first: str,
    second: str,
) -> float:

    return SequenceMatcher(
        None,
        normalize_text(first),
        normalize_text(second),
    ).ratio()


def are_duplicates(
    first: dict,
    second: dict,
) -> bool:

    if first["file"] != second["file"]:
        return False

    # Agents frequently choose slightly different lines
    # for the same issue.
    if (
        first["line"]
        and second["line"]
        and abs(first["line"] - second["line"]) > 5
    ):
        return False

    title_similarity = text_similarity(
        first["title"],
        second["title"],
    )

    message_similarity = text_similarity(
        first["message"],
        second["message"],
    )

    return (
        title_similarity >= 0.72
        or message_similarity >= 0.78
    )


def choose_better_finding(
    first: dict,
    second: dict,
) -> dict:

    first_priority = SEVERITY_PRIORITY.get(
        first["severity"],
        99,
    )

    second_priority = SEVERITY_PRIORITY.get(
        second["severity"],
        99,
    )

    if second_priority < first_priority:
        winner = second.copy()
        loser = first
    elif first_priority < second_priority:
        winner = first.copy()
        loser = second
    elif (
        second["confidence"]
        > first["confidence"]
    ):
        winner = second.copy()
        loser = first
    else:
        winner = first.copy()
        loser = second

    agents = set(
        winner.get("agents", [winner["agent"]])
    )

    agents.add(loser["agent"])

    winner["agents"] = sorted(agents)

    return winner


def deduplicate_findings(
    findings: list[dict],
) -> list[dict]:

    unique: list[dict] = []

    for finding in findings:

        duplicate_index = None

        for index, existing in enumerate(unique):

            if are_duplicates(
                existing,
                finding,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:

            finding = finding.copy()

            finding["agents"] = [
                finding["agent"]
            ]

            unique.append(finding)

        else:

            unique[duplicate_index] = (
                choose_better_finding(
                    unique[duplicate_index],
                    finding,
                )
            )

    return unique


# ---------------------------------------------------------------------
# Prioritization
# ---------------------------------------------------------------------

def prioritize_findings(
    findings: list[dict],
) -> list[dict]:

    findings.sort(
        key=lambda finding: (
            SEVERITY_PRIORITY.get(
                finding["severity"],
                99,
            ),
            -finding["confidence"],
        )
    )

    selected = []
    low_count = 0

    for finding in findings:

        if (
            finding["severity"]
            in {"low", "info"}
        ):

            if (
                low_count
                >= MAX_LOW_SEVERITY_FINDINGS
            ):
                continue

            low_count += 1

        selected.append(finding)

        if len(selected) >= MAX_FINDINGS:
            break

    return selected


# ---------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------

def make_node(
    agent_name: str,
    get_prompt,
):

    def node(state: GraphState) -> dict:

        prompt = (
            get_prompt(state)
            if callable(get_prompt)
            else get_prompt
        )

        response = (
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": state["diff"],
                    },
                ],
                temperature=0,
            )
        )

        items = parse_json_response(
            response.choices[0].message.content
        )

        findings = []

        for item in items:

            normalized = normalize_finding(
                item,
                agent_name,
            )

            if (
                normalized
                and passes_confidence_threshold(
                    normalized
                )
            ):
                findings.append(normalized)

        return {
            "findings": findings
        }

    return node


# ---------------------------------------------------------------------
# Style agent
# ---------------------------------------------------------------------

def _style_prompt(
    state: GraphState,
) -> str:

    patterns_str = (
        "\n".join(state["patterns"])
        if state["patterns"]
        else "None"
    )

    return f"""
{COMMON_RULES}

You are a senior code readability and maintainability reviewer.

Previously observed team patterns:
{patterns_str}

Review for:
- confusing control flow
- overly complex functions
- duplicated business logic
- misleading names
- hidden side effects
- functions with too many responsibilities
- poor abstractions
- inconsistent error-handling patterns
- code that is unnecessarily difficult to maintain

DO NOT report:
- harmless whitespace
- blank-line preferences
- import ordering
- formatter issues
- list[] versus typing.List
- minor personal style preferences

Those belong to automated tools such as formatters and linters,
not an AI code reviewer.
"""


# ---------------------------------------------------------------------
# Merge node
# ---------------------------------------------------------------------

def merge_node(
    state: GraphState,
) -> dict:

    findings = state["findings"]

    deduplicated = (
        deduplicate_findings(findings)
    )

    prioritized = (
        prioritize_findings(deduplicated)
    )

    return {
        "findings": prioritized
    }


# ---------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------

def fan_out(
    state: GraphState,
):

    return [
        Send(
            "static_analysis",
            state,
        ),
        Send(
            "security",
            state,
        ),
        Send(
            "style",
            state,
        ),
        Send(
            "architecture",
            state,
        ),
    ]


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

def build_graph() -> StateGraph:

    builder = StateGraph(GraphState)

    builder.add_node(
        "static_analysis",
        make_node(
            "static_analysis",
            PROMPTS["static_analysis"],
        ),
    )

    builder.add_node(
        "security",
        make_node(
            "security",
            PROMPTS["security"],
        ),
    )

    builder.add_node(
        "style",
        make_node(
            "style",
            _style_prompt,
        ),
    )

    builder.add_node(
        "architecture",
        make_node(
            "architecture",
            PROMPTS["architecture"],
        ),
    )

    builder.add_node(
        "merge",
        merge_node,
    )

    builder.set_conditional_entry_point(
        fan_out
    )

    for name in (
        "static_analysis",
        "security",
        "style",
        "architecture",
    ):
        builder.add_edge(
            name,
            "merge",
        )

    builder.add_edge(
        "merge",
        END,
    )

    return builder.compile()