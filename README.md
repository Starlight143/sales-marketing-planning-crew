# Sales Marketing Planning Crew

AI-powered Sales & Marketing Planning System with Structured Outputs.

This project is part of a broader multi-agent planning framework, adapted from QuantSaaS_Crew for go-to-market and sales strategy use cases.

## What Problem Does This Solve?

Most teams need a fast, structured way to turn an idea or codebase into a concrete sales and marketing plan. This tool creates a usable plan with positioning, target segments, experiments, risks, and next steps in one run.

## Who Is This For?

- Founders and builders validating a new product
- PMs and ops teams preparing a launch plan
- Growth and marketing teams who need structured experiments
- Technical teams who want a GTM plan from a repo scan

## Example Use Cases

- Turn a one-paragraph idea into a GTM plan with measurable experiments
- Scan a project folder and generate positioning + channel recommendations
- Produce a risk-and-mitigation list for a new product launch

## How It Works

It supports two modes:

1) Idea expansion: you provide an idea, the crew expands it into a marketing + sales plan.
2) Project path analysis: you provide a project folder path, choose quick or full scan, and the crew proposes a plan.

Quick scan reads the tree (depth 3) and key files only. Full scan reads all readable text files and their full content.
Sensitive files (keys, .env, certificates) are skipped from content and noted in the scan output.
For stability, full scan enforces a total content budget; if a project is very large, some files may be trimmed or skipped with a note.
Binary/large asset types (images, archives, executables, media, DB files) are skipped from content to preserve quality.

## Output Example (JSON / Markdown)

Structured JSON (excerpt):

```json
{
  "summary": "B2B analytics tool for ops teams",
  "positioning": "Cut reporting time by 60% with automated metrics",
  "experiments": [
    {
      "hypothesis": "Ops teams will book a demo for KPI dashboards",
      "metric": "demo_bookings_per_week",
      "duration_days": 14
    }
  ],
  "risks": [
    "Data access friction slows onboarding"
  ]
}
```

Human-readable Markdown (excerpt):

```md
# GTM Plan
## Positioning
- Cut reporting time by 60% with automated metrics
## Experiments
- KPI demo landing page → measure demo bookings
```

## What This Is NOT

- Not a replacement for human judgment
- Not an automated sales bot
- Not a consulting service

## Setup

1) Install dependencies (Python 3.10+ recommended):

```bash
pip install crewai langchain_openai pydantic
```

2) Add your OpenRouter API key:

Create `openrouter_key.txt` in this folder with a single line:

```
YOUR_OPENROUTER_API_KEY
```

## Run

From this folder:

```bash
python sales_marketing_crew.py
```

Dry run (scan only, no LLM call):

```bash
python sales_marketing_crew.py --dry-run
```

Dry run still uses the interactive menu. Choose mode 2 and provide a project path to see the scan output.

## Output

Each run creates a timestamped folder in `saved_projects/` with:

- `analysis_result.json` (structured marketing plan)
- `README.md` (human-readable report)
