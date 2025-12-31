from __future__ import annotations
import argparse
import json
import re
import os
import signal
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

# Windows compatibility for missing Unix signals used by CrewAI
if sys.platform.startswith("win"):
    _missing = {
        "SIGHUP": 1,
        "SIGUSR1": 10,
        "SIGUSR2": 12,
        "SIGCHLD": 17,
        "SIGPIPE": 13,
        "SIGALRM": 14,
        "SIGTSTP": 20,
        "SIGQUIT": 3,
        "SIGTRAP": 5,
        "SIGABRT": 6,
        "SIGCONT": 18,
        "SIGSTOP": 19,
        "SIGTTIN": 21,
        "SIGTTOU": 22,
        "SIGURG": 23,
        "SIGXCPU": 24,
        "SIGXFSZ": 25,
        "SIGVTALRM": 26,
        "SIGPROF": 27,
        "SIGWINCH": 28,
    }
    for _name, _val in _missing.items():
        if not hasattr(signal, _name):
            setattr(signal, _name, _val)


def to_json_str(obj: Any) -> str:
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(indent=2)
    if hasattr(obj, "json"):
        try:
            return obj.json(indent=2, ensure_ascii=False)
        except TypeError:
            pass
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


class Experiment(BaseModel):
    goal: str = Field(..., description="Experiment goal")
    criteria: str = Field(..., description="Success criteria")


class PlanStep(BaseModel):
    phase: str = Field(..., description="Phase label")
    actions: List[str] = Field(..., description="Concrete actions")
    success_metrics: List[str] = Field(..., description="Success metrics for this phase")


class MarketingPlan(BaseModel):
    project_name: str = Field(..., description="Short snake_case project name")
    mode_used: str = Field(..., description="idea or project_path")
    summary: str = Field(..., description="Overall plan summary")
    target_segments: List[str] = Field(..., description="Primary target segments")
    value_proposition: str = Field(..., description="Core value proposition")
    positioning: str = Field(..., description="Positioning statement")
    channels: List[str] = Field(..., description="Acquisition and marketing channels")
    sales_motion: str = Field(..., description="Sales motion and outreach approach")
    pricing_strategy: str = Field(..., description="Pricing strategy and packaging")
    funnel_steps: List[str] = Field(..., description="Funnel stages")
    plan_steps: List[PlanStep] = Field(..., description="Phased plan with actions and metrics")
    experiments: List[Experiment] = Field(..., description="Validation experiments")
    risks: List[str] = Field(..., description="Key risks and mitigations")
    metrics: List[str] = Field(..., description="North-star and supporting metrics")
    assumptions: List[str] = Field(..., description="Assumptions to validate")


def extract_marketing_plan(result: Any) -> Optional[MarketingPlan]:
    if isinstance(result, MarketingPlan):
        return result
    if isinstance(result, dict):
        try:
            return MarketingPlan(**result)
        except Exception:
            return None

    if hasattr(result, "pydantic") and isinstance(getattr(result, "pydantic", None), MarketingPlan):
        return result.pydantic

    if hasattr(result, "tasks_output") and result.tasks_output:
        for t in result.tasks_output:
            if hasattr(t, "pydantic") and isinstance(getattr(t, "pydantic", None), MarketingPlan):
                return t.pydantic
            for attr in ("json_dict", "output", "raw"):
                if not hasattr(t, attr):
                    continue
                val = getattr(t, attr)
                if isinstance(val, dict):
                    d = val
                elif isinstance(val, str):
                    d = _extract_first_json_object(val)
                else:
                    try:
                        d = json.loads(json.dumps(val, ensure_ascii=False))
                    except Exception:
                        d = None
                if isinstance(d, dict) and "project_name" in d and "summary" in d and "mode_used" in d:
                    try:
                        return MarketingPlan(**d)
                    except Exception:
                        pass

    if hasattr(result, "raw"):
        raw = result.raw
        if isinstance(raw, dict):
            d = raw
        elif isinstance(raw, str):
            d = _extract_first_json_object(raw)
        else:
            try:
                d = json.loads(json.dumps(raw, ensure_ascii=False))
            except Exception:
                d = None
        if isinstance(d, dict) and "project_name" in d and "summary" in d and "mode_used" in d:
            try:
                return MarketingPlan(**d)
            except Exception:
                pass
    return None


def load_api_key() -> str:
    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openrouter_key.txt")
    try:
        with open(key_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if not key or key.lower().startswith("replace_with"):
                raise ValueError("Key file is empty or placeholder")
            return key
    except FileNotFoundError:
        print(f"Error: API key file not found at {key_path}")
        print("Create openrouter_key.txt with your OpenRouter API key.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading API key: {e}")
        sys.exit(1)


def init_llm() -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except Exception as e:
        print("Error: Failed to import langchain_openai/tiktoken.")
        print(f"Details: {e}")
        print("Fix: reinstall dependencies, e.g.:")
        print("  pip install -U --force-reinstall tiktoken langchain-openai")
        sys.exit(1)

    api_key = load_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    return ChatOpenAI(
        model="openai/gpt-5.2",
        temperature=0.7,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "coverage",
    ".pytest_cache",
    "saved_projects",
}
KEY_FILES = {
    "README.md",
    "README.txt",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "Pipfile",
    "setup.py",
    "Dockerfile",
    ".env.example",
    "Makefile",
}
ENTRYPOINT_FILES = {
    "main.py",
    "app.py",
    "server.py",
    "index.py",
    "main.js",
    "server.js",
    "index.js",
}
SENSITIVE_FILENAMES = {
    "openrouter_key.txt",
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.staging",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "client_secret.json",
}
SENSITIVE_EXTS = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".der",
    ".jks",
    ".keystore",
}
SKIP_CONTENT_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
    ".feather",
    ".arrow",
    ".xlsx",
    ".xls",
    ".docx",
    ".pptx",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".flac",
    ".webm",
    ".avi",
    ".mkv",
    ".otf",
    ".ttf",
    ".woff",
    ".woff2",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
}
SENSITIVE_PATTERN = re.compile(
    r"(^|[._-])(api_?key|secret|token|password|passwd|credential|private_key)([._-]|$)"
)
QUICK_MAX_TREE_ENTRIES = 200
QUICK_MAX_DEPTH = 3
QUICK_MAX_FILE_BYTES = 20000
QUICK_MAX_SNIPPET_CHARS = 4000

FULL_MAX_TREE_ENTRIES = None
FULL_MAX_DEPTH = None
FULL_MAX_FILE_BYTES = 1000000
FULL_MAX_SNIPPET_CHARS = None
FULL_MAX_TOTAL_CHARS = 200000


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def fmt_limit(value: Optional[int]) -> str:
    return "unlimited" if value is None else str(value)


def is_sensitive_filename(name: str) -> bool:
    base = name.lower()
    if base in SENSITIVE_FILENAMES:
        return True
    if base.startswith(".env"):
        return True
    if any(base.endswith(ext) for ext in SENSITIVE_EXTS):
        return True
    return bool(SENSITIVE_PATTERN.search(base))


def should_skip_content(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    if ext in SKIP_CONTENT_EXTS:
        return True
    return False


def safe_read_text(
    path: str, max_bytes: Optional[int], max_chars: Optional[int]
) -> Tuple[str, bool, Optional[str]]:
    try:
        with open(path, "rb") as f:
            if max_bytes is None:
                data = f.read()
                truncated = False
            else:
                data = f.read(max_bytes + 1)
                truncated = len(data) > max_bytes
                if truncated:
                    data = data[:max_bytes]
        if b"\x00" in data:
            return "", False, "binary"
        text = data.decode("utf-8", errors="replace")
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        return text, truncated, None
    except Exception:
        return "", False, "error"


def parse_requirements(text: str) -> List[str]:
    deps: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ["==", ">=", "<=", "~=", ">", "<"]:
            if sep in line:
                line = line.split(sep, 1)[0].strip()
                break
        if line and line not in deps:
            deps.append(line)
    return deps


def parse_package_json(text: str) -> List[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    deps: List[str] = []
    for key in ("dependencies", "devDependencies"):
        block = data.get(key, {})
        if isinstance(block, dict):
            for dep in block.keys():
                if dep not in deps:
                    deps.append(dep)
    return deps


def detect_stack(ext_counts: Dict[str, int], deps: List[str]) -> List[str]:
    stack: List[str] = []
    if ext_counts.get("py"):
        stack.append("Python")
    if ext_counts.get("js") or ext_counts.get("ts"):
        stack.append("JavaScript/TypeScript")
    if ext_counts.get("html") or ext_counts.get("css"):
        stack.append("Web")

    dep_set = {d.lower() for d in deps}
    if "django" in dep_set:
        stack.append("Django")
    if "flask" in dep_set:
        stack.append("Flask")
    if "fastapi" in dep_set:
        stack.append("FastAPI")
    if "streamlit" in dep_set:
        stack.append("Streamlit")
    if "react" in dep_set or "next" in dep_set:
        stack.append("React/Next")
    if "vue" in dep_set:
        stack.append("Vue")
    if "express" in dep_set:
        stack.append("Express")
    if "nestjs" in dep_set:
        stack.append("NestJS")
    return sorted(set(stack))


def build_project_context(
    project_path: str,
    max_depth: Optional[int],
    max_tree_entries: Optional[int],
    read_all_files: bool,
    max_file_bytes: Optional[int],
    max_snippet_chars: Optional[int],
    max_total_chars: Optional[int],
) -> Dict[str, Any]:
    root = os.path.abspath(project_path)
    tree_lines: List[str] = [f"{os.path.basename(root)}/"]
    ext_counts: Dict[str, int] = {}
    key_file_snippets: Dict[str, Tuple[str, bool]] = {}
    entry_snippets: Dict[str, Tuple[str, bool]] = {}
    all_file_snippets: List[Tuple[str, str, bool]] = []
    deps: List[str] = []
    notes: List[str] = []
    skipped_binary = 0
    skipped_error = 0
    skipped_sensitive = 0
    skipped_by_ext = 0
    skipped_budget = 0
    budget_trimmed = 0
    total_chars = 0
    truncated_files = 0
    context_truncated = False
    truncate_reasons: List[str] = []

    entry_count = 0
    for current_root, dirs, files in os.walk(root):
        rel_root = os.path.relpath(current_root, root)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if max_depth is not None and depth > max_depth:
            if "max_depth" not in truncate_reasons:
                truncate_reasons.append("max_depth")
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        indent = "  " * depth
        if rel_root != ".":
            tree_lines.append(f"{indent}{os.path.basename(current_root)}/")
            entry_count += 1
            if max_tree_entries is not None and entry_count >= max_tree_entries:
                notes.append(f"Tree truncated after {max_tree_entries} entries.")
                if "max_tree_entries" not in truncate_reasons:
                    truncate_reasons.append("max_tree_entries")
                break

        def _file_priority(n: str) -> Tuple[int, str]:
            if n in KEY_FILES:
                return (0, n)
            if n in ENTRYPOINT_FILES:
                return (1, n)
            return (2, n)

        for name in sorted(files, key=_file_priority):
            if max_tree_entries is not None and entry_count >= max_tree_entries:
                notes.append(f"Tree truncated after {max_tree_entries} entries.")
                if "max_tree_entries" not in truncate_reasons:
                    truncate_reasons.append("max_tree_entries")
                break
            rel_path = os.path.join(rel_root, name) if rel_root != "." else name
            tree_lines.append(f"{indent}  {name}")
            entry_count += 1

            ext = os.path.splitext(name)[1].lstrip(".").lower()
            if ext:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            is_sensitive = is_sensitive_filename(name)
            skip_by_ext = should_skip_content(name)
            if is_sensitive:
                skipped_sensitive += 1
            if skip_by_ext:
                skipped_by_ext += 1

            file_text = None
            file_truncated = False
            file_skip_reason = None
            if read_all_files and not is_sensitive and not skip_by_ext:
                if max_total_chars is not None and total_chars >= max_total_chars:
                    skipped_budget += 1
                    if "max_total_chars" not in truncate_reasons:
                        truncate_reasons.append("max_total_chars")
                    continue
                file_text, file_truncated, file_skip_reason = safe_read_text(
                    os.path.join(current_root, name), max_file_bytes, max_snippet_chars
                )
                if file_skip_reason == "binary":
                    skipped_binary += 1
                elif file_skip_reason == "error":
                    skipped_error += 1
                elif file_text:
                    if max_total_chars is not None:
                        remaining = max_total_chars - total_chars
                        if remaining <= 0:
                            skipped_budget += 1
                            if "max_total_chars" not in truncate_reasons:
                                truncate_reasons.append("max_total_chars")
                            continue
                        if len(file_text) > remaining:
                            file_text = file_text[:remaining]
                            file_truncated = True
                            budget_trimmed += 1
                            if "max_total_chars" not in truncate_reasons:
                                truncate_reasons.append("max_total_chars")
                    all_file_snippets.append((rel_path, file_text, file_truncated))
                    total_chars += len(file_text)
                    if file_truncated:
                        truncated_files += 1

            if name in KEY_FILES:
                if read_all_files:
                    if file_text:
                        if name == "requirements.txt":
                            deps.extend(parse_requirements(file_text))
                        if name == "package.json":
                            deps.extend(parse_package_json(file_text))
                elif len(key_file_snippets) < 5 and not is_sensitive and not skip_by_ext:
                    text, truncated, skip_reason = safe_read_text(
                        os.path.join(current_root, name), max_file_bytes, max_snippet_chars
                    )
                    if skip_reason == "binary":
                        skipped_binary += 1
                    elif skip_reason == "error":
                        skipped_error += 1
                    elif text:
                        key_file_snippets[rel_path] = (text, truncated)
                        if name == "requirements.txt":
                            deps.extend(parse_requirements(text))
                        if name == "package.json":
                            deps.extend(parse_package_json(text))
                        if truncated:
                            truncated_files += 1

            if (
                name in ENTRYPOINT_FILES
                and not read_all_files
                and len(entry_snippets) < 3
                and not is_sensitive
                and not skip_by_ext
            ):
                text, truncated, skip_reason = safe_read_text(
                    os.path.join(current_root, name), max_file_bytes, max_snippet_chars
                )
                if skip_reason == "binary":
                    skipped_binary += 1
                elif skip_reason == "error":
                    skipped_error += 1
                elif text:
                    entry_snippets[rel_path] = (text, truncated)
                    if truncated:
                        truncated_files += 1

        if max_tree_entries is not None and entry_count >= max_tree_entries:
            break

    if not tree_lines:
        notes.append("No files found in project root.")
    if skipped_by_ext:
        notes.append(f"Skipped files by extension: {skipped_by_ext}.")
    if skipped_sensitive:
        notes.append(f"Skipped sensitive files: {skipped_sensitive}.")
    if skipped_budget:
        notes.append(f"Skipped files due to content budget: {skipped_budget}.")
    if budget_trimmed:
        notes.append(f"Trimmed files due to content budget: {budget_trimmed}.")
    if skipped_binary:
        notes.append(f"Skipped binary files: {skipped_binary}.")
    if skipped_error:
        notes.append(f"Skipped unreadable files: {skipped_error}.")
    if truncated_files:
        notes.append(f"Truncated file contents: {truncated_files}.")

    if read_all_files:
        scan_summary = "full scan, all readable text files included"
    else:
        depth_label = str(max_depth) if max_depth is not None else "unlimited"
        entries_label = str(max_tree_entries) if max_tree_entries is not None else "unlimited"
        scan_summary = f"quick scan (depth <= {depth_label}, max {entries_label} entries)"

    context = {
        "root": root,
        "tree": "\n".join(tree_lines),
        "ext_counts": ext_counts,
        "key_files": key_file_snippets,
        "entrypoints": entry_snippets,
        "all_files": all_file_snippets,
        "deps": sorted(set(deps)),
        "stack_guess": detect_stack(ext_counts, deps),
        "notes": notes,
        "scan_summary": scan_summary,
        "context_truncated": bool(truncate_reasons or truncated_files or skipped_budget or budget_trimmed),
        "context_truncate_reasons": truncate_reasons,
        "content_chars": total_chars,
    }
    return context


def format_project_context(context: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append("=== PROJECT METADATA ===")
    parts.append(f"Project path: {context['root']}")
    parts.append(f"Scan summary: {context.get('scan_summary', 'scan')}")
    if context.get("context_truncated"):
        reasons = context.get("context_truncate_reasons", [])
        if reasons:
            parts.append(f"Context truncated: yes ({', '.join(reasons)})")
        else:
            parts.append("Context truncated: yes")
    else:
        parts.append("Context truncated: no")
    parts.append(f"Content chars included: {context.get('content_chars', 0)}")

    parts.append("\n=== PROJECT TREE ===")
    parts.append(context["tree"])

    ext_counts = context.get("ext_counts", {})
    if ext_counts:
        parts.append("\n=== EXTENSION COUNTS ===")
        for ext, count in sorted(ext_counts.items(), key=lambda x: (-x[1], x[0])):
            parts.append(f"- {ext}: {count}")

    stack_guess = context.get("stack_guess", [])
    if stack_guess:
        parts.append("\n=== STACK GUESS ===")
        parts.append(", ".join(stack_guess))

    deps = context.get("deps", [])
    if deps:
        parts.append("\n=== DEPENDENCIES (SAMPLED) ===")
        parts.append(", ".join(deps[:40]))

    all_files = context.get("all_files", [])
    if all_files:
        parts.append("\n=== FILE CONTENTS ===")
        for path, text, truncated in all_files:
            suffix = " (truncated)" if truncated else ""
            parts.append(f"[{path}]{suffix}\n{text}")
    else:
        key_files = context.get("key_files", {})
        if key_files:
            parts.append("\n=== KEY FILE EXCERPTS ===")
            for path, (text, truncated) in key_files.items():
                suffix = " (truncated)" if truncated else ""
                parts.append(f"[{path}]{suffix}\n{text}")

        entrypoints = context.get("entrypoints", {})
        if entrypoints:
            parts.append("\n=== ENTRYPOINT EXCERPTS ===")
            for path, (text, truncated) in entrypoints.items():
                suffix = " (truncated)" if truncated else ""
                parts.append(f"[{path}]{suffix}\n{text}")

    notes = context.get("notes", [])
    if notes:
        parts.append("\n=== NOTES ===")
        for note in notes:
            parts.append(f"- {note}")

    return "\n".join(parts)


COMMON_OUTPUT_RULES = """
Output format:
- Key insights (max 10 bullets)
- Assumptions / required inputs (max 5 bullets)
- Confidence: Low/Medium/High + validation method

Rules:
- Be concrete and specific.
- Do not reference other roles or their output.
"""

ARBITER_OUTPUT_RULES = """
Output sections only:
[Consensus]
- Target segments: ...
- Value proposition: ...
- Positioning: ...
- Channels: ...
- Sales motion: ...
- Pricing strategy: ...
- Funnel steps: ...
- Metrics: ...
- Risks: ...
- Assumptions: ...

[Disagreement]
- bullets only

[Plan]
- Phase 1: ...
- Phase 2: ...
- Phase 3: ...

[Experiments]
- Each item: Goal | Success Criteria
"""


def build_crew(user_problem: str, mode_used: str, language_hint: str, llm: Any) -> Crew:
    base_desc = f"""
Task:
{user_problem}

Language: {language_hint}
{COMMON_OUTPUT_RULES}
"""

    research = Agent(
        role="Research",
        goal="Define target segments, market context, and demand drivers.",
        backstory="Market researcher focused on ICP definition and market sizing.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    growth = Agent(
        role="Growth",
        goal="Propose acquisition channels, growth loops, and messaging angles.",
        backstory="Growth strategist focused on channels and distribution.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    sales = Agent(
        role="Sales",
        goal="Design sales motion, outreach, pipeline, and conversion tactics.",
        backstory="Sales leader focused on pipeline design and conversion.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    product = Agent(
        role="Product",
        goal="Clarify value proposition, packaging, pricing, and differentiation.",
        backstory="Product strategist focused on positioning and pricing.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    critic = Agent(
        role="Critic",
        goal="Challenge assumptions and highlight gaps or risks in the plan.",
        backstory="Skeptical analyst focused on validation risks.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    arbiter = Agent(
        role="Arbiter",
        goal="Synthesize a single plan without adding new ideas.",
        backstory=ARBITER_OUTPUT_RULES,
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    format_checker = Agent(
        role="Format Checker",
        goal="Convert Arbiter output into a valid MarketingPlan JSON object.",
        backstory=(
            "You are a strict JSON formatter. "
            "Do not add new info. Only structure the Arbiter content. "
            "Generate a short snake_case project_name. "
            f"Set mode_used to '{mode_used}'. "
            "If a field is missing, use 'TBD' for strings and ['TBD'] for lists."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    t_research = Task(description=base_desc, agent=research, expected_output="Marketing insights")
    t_growth = Task(description=base_desc, agent=growth, expected_output="Channel strategy")
    t_sales = Task(description=base_desc, agent=sales, expected_output="Sales plan")
    t_product = Task(description=base_desc, agent=product, expected_output="Positioning and pricing")
    t_critic = Task(description=base_desc, agent=critic, expected_output="Risks and gaps")

    t_arbiter = Task(
        description=(
            "Synthesize a single plan using the inputs. "
            "No new ideas. Use only the output sections in ARBITER_OUTPUT_RULES. "
            "Follow the labeled bullets in the Consensus section.\n"
            f"Language: {language_hint}"
        ),
        agent=arbiter,
        context=[t_research, t_growth, t_sales, t_product, t_critic],
        expected_output="Consensus, disagreement, plan, experiments",
    )

    t_format = Task(
        description=(
            "Convert Arbiter report into the MarketingPlan JSON schema. "
            "Fill all required fields. Use concise lists. "
            "Project_name should be snake_case and short."
        ),
        agent=format_checker,
        context=[t_arbiter],
        output_pydantic=MarketingPlan,
        expected_output="Valid MarketingPlan JSON",
    )

    return Crew(
        agents=[research, growth, sales, product, critic, arbiter, format_checker],
        tasks=[t_research, t_growth, t_sales, t_product, t_critic, t_arbiter, t_format],
        process=Process.sequential,
        verbose=True,
    )


def sanitize_name(name: str) -> str:
    safe = "".join([c for c in name if c.isalnum() or c in ("_", "-")]).strip()
    return safe or "project"


def save_project_output(result: Optional[MarketingPlan], mode_used: str) -> None:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_projects")
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = result.project_name if result and result.project_name else f"{mode_used}_plan"
    safe_name = sanitize_name(project_name)
    project_dir = os.path.join(base_dir, f"{timestamp}_{safe_name}")
    os.makedirs(project_dir, exist_ok=True)

    json_path = os.path.join(project_dir, "analysis_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(to_json_str(result))

    md_path = os.path.join(project_dir, "README.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {project_name}\n\n")
        f.write(f"- Date: {timestamp}\n")
        f.write(f"- Mode: {mode_used}\n\n")
        if result:
            f.write(f"## Summary\n{result.summary}\n\n")
            f.write("## Target Segments\n")
            for item in result.target_segments:
                f.write(f"- {item}\n")
            f.write("\n## Value Proposition\n")
            f.write(f"{result.value_proposition}\n\n")
            f.write("## Positioning\n")
            f.write(f"{result.positioning}\n\n")
            f.write("## Channels\n")
            for item in result.channels:
                f.write(f"- {item}\n")
            f.write("\n## Sales Motion\n")
            f.write(f"{result.sales_motion}\n\n")
            f.write("## Pricing Strategy\n")
            f.write(f"{result.pricing_strategy}\n\n")
            f.write("## Funnel Steps\n")
            for item in result.funnel_steps:
                f.write(f"- {item}\n")
            f.write("\n## Plan Steps\n")
            for step in result.plan_steps:
                f.write(f"- {step.phase}\n")
                for action in step.actions:
                    f.write(f"  - {action}\n")
                for metric in step.success_metrics:
                    f.write(f"  - Metric: {metric}\n")
            f.write("\n## Experiments\n")
            for exp in result.experiments:
                f.write(f"- Goal: {exp.goal}\n")
                f.write(f"  - Criteria: {exp.criteria}\n")
            f.write("\n## Risks\n")
            for item in result.risks:
                f.write(f"- {item}\n")
            f.write("\n## Metrics\n")
            for item in result.metrics:
                f.write(f"- {item}\n")
            f.write("\n## Assumptions\n")
            for item in result.assumptions:
                f.write(f"- {item}\n")
        else:
            f.write("\n(No structured report produced.)\n")

    print(f"\n[System] Project saved to: {project_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales & Marketing Planning Crew")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan project and print context without calling the LLM.",
    )
    args = parser.parse_args()

    print("==========================================")
    print("   Sales & Marketing Planning Crew        ")
    print("==========================================")

    while True:
        print("\nSelect Mode:")
        print("1) Idea expansion (marketing + sales plan)")
        print("2) Project path analysis (auto-detect)")
        mode_input = input("Enter 1 or 2 [Default: 1]: ").strip()
        if mode_input in ("", "1"):
            mode_used = "idea"
            break
        if mode_input == "2":
            mode_used = "project_path"
            break
        print("Invalid selection. Try again.")

    project_context_text = ""
    extra_notes = ""
    if mode_used == "project_path":
        project_path = input("\nEnter project folder path:\n> ").strip()
        if not project_path or not os.path.isdir(project_path):
            print("Error: Invalid project path.")
            sys.exit(1)
        while True:
            print("\nSelect scan depth:")
            print("1) Quick scan (depth 3, key files only)")
            print("2) Full scan (all files, full content)")
            scan_input = input("Enter 1 or 2 [Default: 1]: ").strip()
            if scan_input in ("", "1"):
                scan_mode = "quick"
                break
            if scan_input == "2":
                scan_mode = "full"
                break
            print("Invalid selection. Try again.")
        extra_notes = input("\nOptional: add context about product/market (press Enter to skip):\n> ").strip()
        if scan_mode == "full":
            context = build_project_context(
                project_path,
                FULL_MAX_DEPTH,
                FULL_MAX_TREE_ENTRIES,
                True,
                FULL_MAX_FILE_BYTES,
                FULL_MAX_SNIPPET_CHARS,
                FULL_MAX_TOTAL_CHARS,
            )
            settings_line = (
                "Operator settings: "
                f"scan_mode=full, max_depth={fmt_limit(FULL_MAX_DEPTH)}, "
                f"max_tree_entries={fmt_limit(FULL_MAX_TREE_ENTRIES)}, "
                f"max_file_bytes={fmt_limit(FULL_MAX_FILE_BYTES)}, "
                f"max_total_chars={fmt_limit(FULL_MAX_TOTAL_CHARS)}"
            )
        else:
            context = build_project_context(
                project_path,
                QUICK_MAX_DEPTH,
                QUICK_MAX_TREE_ENTRIES,
                False,
                QUICK_MAX_FILE_BYTES,
                QUICK_MAX_SNIPPET_CHARS,
                None,
            )
            settings_line = (
                "Operator settings: "
                f"scan_mode=quick, max_depth={fmt_limit(QUICK_MAX_DEPTH)}, "
                f"max_tree_entries={fmt_limit(QUICK_MAX_TREE_ENTRIES)}, "
                f"max_file_bytes={fmt_limit(QUICK_MAX_FILE_BYTES)}, "
                "max_total_chars=unlimited"
            )
        project_context_text = format_project_context(context)
        if args.dry_run:
            print("\n\n################################################")
            print("## DRY RUN: PROJECT CONTEXT (NO LLM CALL) ##")
            print("################################################\n")
            print(project_context_text)
            return
        user_problem = (
            "Analyze this project and produce a sales and marketing plan.\n\n"
            f"{settings_line}\n\n"
            f"{project_context_text}\n\n"
        )
        if extra_notes:
            user_problem += f"User notes: {extra_notes}\n"
    else:
        if args.dry_run:
            print("Error: --dry-run requires project path mode.")
            sys.exit(1)
        idea = input("\nDescribe your idea or plan:\n> ").strip()
        if not idea:
            print("Error: Empty input.")
            sys.exit(1)
        extra_notes = input("\nOptional: add target market or constraints (press Enter to skip):\n> ").strip()
        user_problem = f"Idea: {idea}\n"
        if extra_notes:
            user_problem += f"Notes: {extra_notes}\n"

    language_hint = "Traditional Chinese" if contains_cjk(user_problem) else "English"
    llm = init_llm()
    crew = build_crew(user_problem, mode_used=mode_used, language_hint=language_hint, llm=llm)

    try:
        result = crew.kickoff()
        plan = extract_marketing_plan(result)

        print("\n\n################################################")
        print("## FINAL STRUCTURED OUTPUT (JSON Format) ##")
        print("################################################\n")
        if plan:
            print(to_json_str(plan))
        else:
            print("[Warn] MarketingPlan not parsed.")
            sys.exit(1)

        save_project_output(plan, mode_used=mode_used)
    except Exception as e:
        print(f"\n[Error] An error occurred during execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
