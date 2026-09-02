import time
from collections import Counter

import httpx
import jwt
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from models import PullRequest, ReviewRequest, Settings


settings = Settings()

engine = create_async_engine(settings.database_url)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

app = FastAPI()

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

INLINE_SEVERITIES = {
    "critical",
    "high",
    "medium",
}

SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

SEVERITY_ICON = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "info": "⚪",
}


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
    }


# ---------------------------------------------------------------------
# Finding helpers
# ---------------------------------------------------------------------

def _normalize_severity(value: str) -> str:
    severity = str(value or "info").lower().strip()

    mapping = {
        "error": "high",
        "warning": "medium",
        "style": "low",
    }

    severity = mapping.get(
        severity,
        severity,
    )

    if severity not in SEVERITY_ORDER:
        return "info"

    return severity


def _finding_agents(finding: dict) -> list[str]:
    agents = finding.get("agents")

    if isinstance(agents, list) and agents:
        return [
            str(agent)
            for agent in agents
            if agent
        ]

    agent = finding.get("agent")

    if agent:
        return [str(agent)]

    return []


def _confidence_percentage(finding: dict) -> int | None:
    confidence = finding.get("confidence")

    if confidence is None:
        return None

    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None

    value = max(
        0.0,
        min(value, 1.0),
    )

    return round(value * 100)


def _finding_title(finding: dict) -> str:
    title = str(
        finding.get("title")
        or ""
    ).strip()

    if title:
        return title

    message = str(
        finding.get("message")
        or "Code review finding"
    ).strip()

    if len(message) <= 80:
        return message

    return (
        message[:77].rstrip()
        + "..."
    )


def _sort_findings(findings: list[dict]) -> list[dict]:
    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER.get(
                _normalize_severity(
                    finding.get("severity")
                ),
                99,
            ),
            -(
                float(
                    finding.get("confidence", 0)
                    or 0
                )
            ),
        ),
    )


# ---------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------

def _build_summary(
    findings: list[dict],
) -> str:

    findings = _sort_findings(findings)

    severity_counts = Counter(
        _normalize_severity(
            finding.get("severity")
        )
        for finding in findings
    )

    lines = [
        "## 🤖 AI Code Review",
        "",
        f"Found **{len(findings)} actionable findings**.",
        "",
        "### Review Summary",
        "",
        "| Severity | Count |",
        "|---|---:|",
        (
            f"| 🔴 Critical | "
            f"{severity_counts.get('critical', 0)} |"
        ),
        (
            f"| 🟠 High | "
            f"{severity_counts.get('high', 0)} |"
        ),
        (
            f"| 🟡 Medium | "
            f"{severity_counts.get('medium', 0)} |"
        ),
        (
            f"| 🔵 Low | "
            f"{severity_counts.get('low', 0)} |"
        ),
        (
            f"| ⚪ Info | "
            f"{severity_counts.get('info', 0)} |"
        ),
        "",
    ]

    high_priority = [
        finding
        for finding in findings
        if _normalize_severity(
            finding.get("severity")
        )
        in {
            "critical",
            "high",
            "medium",
        }
    ]

    if high_priority:
        lines.extend(
            [
                "### Priority Findings",
                "",
            ]
        )

        for finding in high_priority:
            severity = _normalize_severity(
                finding.get("severity")
            )

            icon = SEVERITY_ICON[severity]

            title = _finding_title(
                finding
            )

            file_path = finding.get(
                "file",
                "unknown",
            )

            line = finding.get(
                "line",
                "?",
            )

            confidence = (
                _confidence_percentage(
                    finding
                )
            )

            agents = _finding_agents(
                finding
            )

            lines.append(
                f"- {icon} **"
                f"{severity.upper()}"
                f" — {title}** "
                f"(`{file_path}:{line}`)"
            )

            metadata = []

            if agents:
                metadata.append(
                    "Detected by: "
                    + ", ".join(agents)
                )

            if confidence is not None:
                metadata.append(
                    f"Confidence: {confidence}%"
                )

            if metadata:
                lines.append(
                    "  - "
                    + " | ".join(metadata)
                )

        lines.append("")

    low_priority = [
        finding
        for finding in findings
        if _normalize_severity(
            finding.get("severity")
        )
        in {
            "low",
            "info",
        }
    ]

    if low_priority:
        lines.extend(
            [
                "### Additional Observations",
                "",
            ]
        )

        for finding in low_priority:
            severity = _normalize_severity(
                finding.get("severity")
            )

            icon = SEVERITY_ICON[severity]

            title = _finding_title(
                finding
            )

            file_path = finding.get(
                "file",
                "unknown",
            )

            line = finding.get(
                "line",
                "?",
            )

            lines.append(
                f"- {icon} **{title}** "
                f"(`{file_path}:{line}`)"
            )

        lines.append("")

    lines.extend(
        [
            "---",
            (
                "_Generated by the AI Code Reviewer. "
                "Inline comments contain detailed recommendations "
                "for actionable findings._"
            ),
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Inline comment generation
# ---------------------------------------------------------------------

def _build_inline_comment(
    finding: dict,
) -> str:

    severity = _normalize_severity(
        finding.get("severity")
    )

    icon = SEVERITY_ICON[severity]

    title = _finding_title(
        finding
    )

    message = str(
        finding.get("message")
        or ""
    ).strip()

    agents = _finding_agents(
        finding
    )

    confidence = (
        _confidence_percentage(
            finding
        )
    )

    lines = [
        (
            f"{icon} **{severity.upper()} "
            f"— {title}**"
        ),
        "",
        message,
    ]

    metadata = []

    if agents:
        metadata.append(
            "Detected by: "
            + ", ".join(agents)
        )

    if confidence is not None:
        metadata.append(
            f"Confidence: {confidence}%"
        )

    if metadata:
        lines.extend(
            [
                "",
                "---",
                " | ".join(metadata),
            ]
        )

    return "\n".join(lines)


def _build_inline_comments(
    findings: list[dict],
) -> list[dict]:

    comments = []

    for finding in _sort_findings(
        findings
    ):

        severity = _normalize_severity(
            finding.get("severity")
        )

        if severity not in INLINE_SEVERITIES:
            continue

        file_path = finding.get("file")

        try:
            line = int(
                finding.get("line")
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            line = 0

        if not file_path or line <= 0:
            continue

        comments.append(
            {
                "path": file_path,
                "line": line,
                "side": "RIGHT",
                "body": _build_inline_comment(
                    finding
                ),
            }
        )

    return comments


# ---------------------------------------------------------------------
# GitHub review
# ---------------------------------------------------------------------

@app.post("/post-review")
async def post_review(
    request: ReviewRequest,
):

    if not request.findings:
        await _mark_pr_reviewed(
            request.pr_id
        )

        return {
            "status": "ok",
            "findings": 0,
        }

    token = get_installation_token(
        request.installation_id
    )

    headers = {
        "Authorization": (
            f"Bearer {token}"
        ),
        "Accept": (
            "application/vnd.github+json"
        ),
        "X-GitHub-Api-Version": (
            "2022-11-28"
        ),
    }

    url = (
        "https://api.github.com/repos/"
        f"{request.repo_full_name}"
        f"/pulls/{request.pr_number}"
        "/reviews"
    )

    summary = _build_summary(
        request.findings
    )

    inline_comments = (
        _build_inline_comments(
            request.findings
        )
    )

    payload = {
        "event": "COMMENT",
        "body": summary,
    }

    if inline_comments:
        payload["comments"] = (
            inline_comments
        )

    async with httpx.AsyncClient(
        timeout=30.0
    ) as client:

        response = await client.post(
            url,
            json=payload,
            headers=headers,
        )

        # GitHub can reject inline comments when a line is
        # not part of the current PR diff. In that case,
        # fall back to posting the review summary only.
        if (
            response.status_code == 422
            and inline_comments
        ):

            fallback_payload = {
                "event": "COMMENT",
                "body": summary,
            }

            response = await client.post(
                url,
                json=fallback_payload,
                headers=headers,
            )

        response.raise_for_status()

    await _mark_pr_reviewed(
        request.pr_id
    )

    return {
        "status": "ok",
        "findings": len(
            request.findings
        ),
        "inline_comments": len(
            inline_comments
        ),
    }


# ---------------------------------------------------------------------
# Database state
# ---------------------------------------------------------------------

async def _mark_pr_reviewed(
    pr_id,
) -> None:

    async with (
        AsyncSessionLocal()
        as session
    ):

        await session.execute(
            update(PullRequest)
            .where(
                PullRequest.id
                == pr_id
            )
            .values(
                status="reviewed"
            )
        )

        await session.commit()


# ---------------------------------------------------------------------
# GitHub authentication
# ---------------------------------------------------------------------

def get_installation_token(
    installation_id: int,
) -> str:

    now = int(
        time.time()
    )

    payload = {
        "iat": now - 60,
        "exp": now + 600,
        "iss": settings.github_app_id,
    }

    private_key = (
        settings
        .github_app_private_key
        .replace(
            "\\n",
            "\n",
        )
    )

    encoded_jwt = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
    )

    headers = {
        "Authorization": (
            f"Bearer {encoded_jwt}"
        ),
        "Accept": (
            "application/vnd.github+json"
        ),
        "X-GitHub-Api-Version": (
            "2022-11-28"
        ),
    }

    url = (
        "https://api.github.com/app/"
        f"installations/{installation_id}"
        "/access_tokens"
    )

    with httpx.Client(
        timeout=30.0
    ) as client:

        response = client.post(
            url,
            headers=headers,
        )

        response.raise_for_status()

        data = response.json()

    return data["token"]