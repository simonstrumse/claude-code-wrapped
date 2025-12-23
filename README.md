# Claude Code Wrapped 🎁

Generate a shareable, local-only “Wrapped” report from your Claude Code usage, your local repos, and Git history.

## Highlights

- Sessions, messages, days active, peak day, streaks
- Tokens: **logged + estimate range** (conservative/aggressive)
- Commit momentum + commit-hour activity
- Local codebase stats (projects/files/lines)
- Git history (commits, lines added/deleted)
- GitHub year stats + repos created this year (via `gh`)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/simonstrumse/claude-code-wrapped.git
cd claude-code-wrapped

# Install deps (optional, for auto PNG)
pip install -r requirements.txt

# Run the generator (interactive prompts)
python generate.py
```

You’ll be prompted to:
- Choose a project folder to scan (optional)
- Include GitHub stats and log in with `gh` if needed

Outputs:
- `output/wrapped.html`
- `output/wrapped.png` (if Playwright/Selenium is available)

## Requirements

- Python 3.10+
- Claude Code used locally (`~/.claude/`)
- Optional: Playwright for PNG capture
- Optional: GitHub CLI (`gh`) for GitHub stats

### Playwright setup (recommended)

```bash
pip install playwright
playwright install chromium
```

## Usage

```bash
# Basic usage — reads from ~/.claude/ and prompts for project folder
python generate.py

# Custom output path
python generate.py --output my_wrapped.png

# Just generate HTML
python generate.py --html-only

# Just print stats as JSON
python generate.py --stats-only

# Scan a specific code directory
python generate.py --code-dir ~/Projects

# Max stats scan (includes hidden + nested repos, relaxed excludes)
python generate.py --max-stats

# Include GitHub stats (requires gh auth)
python generate.py --github-stats

# Estimate total tokens across all projects
python generate.py --estimate-tokens --estimate-by-commits

# Redact project names
python generate.py --redact-projects
```

## Token Estimates

When `--estimate-tokens` is enabled, the report shows a **conservative** estimate:
- **Conservative**: line-weighted scaling from known Claude projects to your full codebase

The report also shows the **coverage window** (the date range in `~/.claude/stats-cache.json`).

## Data Sources

| Source | What it provides |
|--------|------------------|
| `~/.claude/stats-cache.json` | Sessions, tokens, daily activity |
| `~/.claude/projects/` | Project names touched in Claude |
| Local filesystem | Codebase file/line counts |
| Git history | Commits + churn metrics |
| `gh` GraphQL API | GitHub contributions + created repos |

**Privacy note:** All processing is local. No data is sent anywhere.

## Customization

- Edit `wrapped.html` to change layout and styling.
- Add stats in `parse_stats.py`, then surface in `wrapped.html` and `generate.py`.

## Credits

Inspired by the original project from [0xleal](https://github.com/0xleal/claude-code-wrapped).
