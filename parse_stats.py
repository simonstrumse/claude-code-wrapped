#!/usr/bin/env python3
"""
Claude Code Wrapped - Stats Parser
Parses ~/.claude/ data to generate wrapped statistics
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class WrappedStats:
    # Core metrics
    total_sessions: int
    total_messages: int
    total_tool_calls: int
    total_days_active: int
    date_range_start: str
    date_range_end: str
    
    # Token stats
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    total_tokens: int
    
    # Model usage
    primary_model: str
    model_percentage: float
    
    # Peak activity
    peak_day_date: str
    peak_day_messages: int
    peak_day_tool_calls: int
    peak_day_sessions: int
    
    # Time patterns
    peak_hour: int
    peak_hour_sessions: int
    hour_distribution: dict
    
    # Longest session
    longest_session_duration_hours: float
    longest_session_messages: int
    longest_session_date: str
    
    # Projects
    projects: list
    project_count: int

    # Local codebase stats
    codebase_repo_count: int
    codebase_projects: list
    codebase_file_count: int
    codebase_line_count: int
    codebase_edit_hour_distribution: dict
    codebase_peak_edit_hour: int
    codebase_peak_edit_hour_files: int

    # Git history stats
    git_repo_count: int
    git_commit_count: int
    git_lines_added: int
    git_lines_deleted: int
    git_commit_hour_distribution: dict
    git_peak_commit_hour: int
    git_peak_commit_hour_commits: int
    
    # Derived fun stats
    tokens_as_pages: int  # ~250 words per page, ~0.75 tokens per word
    coding_personality: str
    streak_days: int


def parse_stats_cache(claude_dir: Path) -> dict:
    """Parse the stats-cache.json file"""
    stats_file = claude_dir / "stats-cache.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Stats cache not found at {stats_file}")
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def parse_history(claude_dir: Path) -> list[dict]:
    """Parse history.jsonl for project breakdown"""
    history_file = claude_dir / "history.jsonl"
    if not history_file.exists():
        return []
    
    entries = []
    with open(history_file, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return entries


def _infer_project_name_from_cwd(cwd: str) -> str:
    """Infer a friendly project name from a working directory path."""
    path = Path(cwd)

    if path.exists():
        # Prefer the git root if present to avoid subdirectory names.
        for parent in [path] + list(path.parents):
            if (parent / ".git").exists():
                return parent.name

    return path.name


def _extract_project_cwd(project_dir: Path) -> Optional[str]:
    """Find a representative cwd from project jsonl files."""
    for file in project_dir.glob("*.jsonl"):
        try:
            with file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cwd = (
                        entry.get("cwd")
                        or entry.get("workingDirectory")
                        or entry.get("projectPath")
                        or entry.get("project_path")
                    )
                    if cwd:
                        return cwd
        except OSError:
            continue
    return None


def _fallback_project_name(project_dir_name: str) -> str:
    """Fallback extraction from encoded directory names."""
    parts = [p for p in project_dir_name.split("-") if p]

    # Find the last meaningful segment (after workspace/personal/etc)
    markers = {"workspace", "projects", "repos", "code", "dev"}
    exclude = {"personal", "work", "talentprotocol", "documents"}

    name_parts = []
    capture = False
    for part in parts:
        lower = part.lower()
        if lower in markers:
            capture = True
            continue
        if capture and lower not in exclude:
            name_parts.append(part)

    if name_parts:
        return "-".join(name_parts)

    return parts[-1] if parts else project_dir_name


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".next",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    "coverage",
    "target",
    "out",
    "bin",
    "obj",
    "Pods",
    "DerivedData",
    ".gradle",
    ".terraform",
    ".turbo",
    ".cache",
    "vendor",
    "tmp",
}

RELAXED_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "Pods",
    "DerivedData",
    "target",
}

EXCLUDED_FILE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".mp4",
    ".mp3",
    ".mov",
    ".avi",
    ".dmg",
    ".jar",
    ".class",
    ".o",
    ".a",
    ".so",
    ".dylib",
    ".exe",
    ".dll",
    ".wasm",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".ico",
    ".icns",
    ".psd",
    ".sketch",
    ".db",
    ".sqlite",
    ".sqlite3",
}

DEFAULT_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024


def _should_skip_dir(name: str, include_hidden: bool, exclude_dirs: set[str]) -> bool:
    if name in exclude_dirs:
        return True
    if not include_hidden and name.startswith("."):
        return True
    return False


def _is_probably_text(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
    except OSError:
        return False

    if b"\x00" in chunk:
        return False

    return True


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
    except OSError:
        return 0


def resolve_exclude_dirs(relax_excludes: bool) -> set[str]:
    return set(RELAXED_EXCLUDED_DIRS if relax_excludes else DEFAULT_EXCLUDED_DIRS)


def find_git_repos(
    code_dir: Optional[Path],
    include_hidden: bool,
    include_nested_repos: bool,
    exclude_dirs: set[str],
) -> list[Path]:
    if code_dir is None or not code_dir.exists():
        return []

    repo_paths = []
    for root, dirs, _ in os.walk(code_dir):
        root_path = Path(root)
        if (root_path / ".git").exists():
            repo_paths.append(root_path)
            if not include_nested_repos:
                dirs[:] = []
                continue

        filtered = []
        for d in dirs:
            if d == ".git":
                continue
            if _should_skip_dir(d, include_hidden, exclude_dirs):
                continue
            filtered.append(d)
        dirs[:] = filtered

    return repo_paths


def generate_git_stats(repo_paths: list[Path]) -> dict:
    if not repo_paths or shutil.which("git") is None:
        return {
            "repo_count": len(repo_paths),
            "commit_count": 0,
            "lines_added": 0,
            "lines_deleted": 0,
            "commit_hour_distribution": {str(i): 0 for i in range(24)},
            "peak_commit_hour": 0,
            "peak_commit_hour_commits": 0,
        }

    commit_count = 0
    lines_added = 0
    lines_deleted = 0
    commit_hour_distribution = {str(i): 0 for i in range(24)}

    for repo in repo_paths:
        cmd = [
            "git",
            "-C",
            str(repo),
            "log",
            "--pretty=format:%ct",
            "--numstat",
            "--no-renames",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
            )
        except OSError:
            continue

        if not proc.stdout:
            continue

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                if line.isdigit():
                    commit_count += 1
                    hour = datetime.fromtimestamp(int(line)).hour
                    commit_hour_distribution[str(hour)] += 1
                continue

            added, deleted, _ = line.split("\t", 2)
            if added.isdigit():
                lines_added += int(added)
            if deleted.isdigit():
                lines_deleted += int(deleted)

        proc.wait()

    if commit_count > 0:
        peak_hour, peak_commits = max(commit_hour_distribution.items(), key=lambda x: x[1])
        peak_hour = int(peak_hour)
    else:
        peak_hour = 0
        peak_commits = 0

    return {
        "repo_count": len(repo_paths),
        "commit_count": commit_count,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "commit_hour_distribution": commit_hour_distribution,
        "peak_commit_hour": peak_hour,
        "peak_commit_hour_commits": peak_commits,
    }


def generate_codebase_stats(
    code_dir: Optional[Path],
    include_hidden: bool,
    max_file_size_bytes: int,
    exclude_dirs: set[str],
) -> dict:
    """Scan a local code directory to estimate repos, files, lines, and edit hours."""
    if code_dir is None or not code_dir.exists():
        return {
            "repo_count": 0,
            "projects": [],
            "file_count": 0,
            "line_count": 0,
            "edit_hour_distribution": {str(i): 0 for i in range(24)},
            "peak_edit_hour": 0,
            "peak_edit_hour_files": 0,
        }

    project_dirs = [
        d for d in code_dir.iterdir()
        if d.is_dir() and (include_hidden or not d.name.startswith("."))
    ]
    project_names = sorted({d.name for d in project_dirs}, key=lambda s: s.lower())
    repo_count = len(project_dirs)
    file_count = 0
    line_count = 0
    edit_hour_distribution = {str(i): 0 for i in range(24)}

    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if not _should_skip_dir(d, include_hidden, exclude_dirs)]

        for filename in files:
            if not include_hidden and filename.startswith("."):
                continue

            path = Path(root) / filename
            if path.suffix.lower() in EXCLUDED_FILE_EXTS:
                continue

            if path.is_symlink():
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            if stat.st_size > max_file_size_bytes:
                continue

            if not _is_probably_text(path):
                continue

            file_count += 1
            line_count += _count_lines(path)

            hour = datetime.fromtimestamp(stat.st_mtime).hour
            edit_hour_distribution[str(hour)] += 1

    if file_count > 0:
        peak_hour, peak_files = max(edit_hour_distribution.items(), key=lambda x: x[1])
        peak_hour = int(peak_hour)
    else:
        peak_hour = 0
        peak_files = 0

    return {
        "repo_count": repo_count,
        "projects": project_names,
        "file_count": file_count,
        "line_count": line_count,
        "edit_hour_distribution": edit_hour_distribution,
        "peak_edit_hour": peak_hour,
        "peak_edit_hour_files": peak_files,
    }


def get_projects(claude_dir: Path) -> list[str]:
    """Get list of projects from projects directory."""
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return []

    projects = []
    for item in projects_dir.iterdir():
        if not item.is_dir():
            continue

        cwd = _extract_project_cwd(item)
        if cwd:
            project_name = _infer_project_name_from_cwd(cwd)
        else:
            project_name = _fallback_project_name(item.name)

        if project_name and project_name not in projects:
            projects.append(project_name)

    return sorted(projects)


def calculate_streak(daily_activity: list[dict]) -> int:
    """Calculate longest streak of consecutive coding days"""
    if not daily_activity:
        return 0
    
    dates = sorted([datetime.strptime(d['date'], '%Y-%m-%d') for d in daily_activity])
    
    max_streak = 1
    current_streak = 1
    
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        elif (dates[i] - dates[i-1]).days > 1:
            current_streak = 1
    
    return max_streak


def determine_personality(hour_counts: dict, total_sessions: int, peak_day_messages: int) -> str:
    """Determine coding personality based on patterns"""
    
    # Convert string keys to int if needed
    hours = {int(k): v for k, v in hour_counts.items()}
    
    # Night owl: most sessions between 10pm-3am
    night_sessions = sum(hours.get(h, 0) for h in [22, 23, 0, 1, 2, 3])
    # Early bird: most sessions between 5am-9am  
    morning_sessions = sum(hours.get(h, 0) for h in [5, 6, 7, 8, 9])
    # Afternoon warrior: most sessions between 2pm-6pm
    afternoon_sessions = sum(hours.get(h, 0) for h in [14, 15, 16, 17, 18])
    
    total = sum(hours.values())
    
    if night_sessions / total > 0.3:
        base = "Night Owl 🦉"
    elif morning_sessions / total > 0.3:
        base = "Early Bird 🐦"
    elif afternoon_sessions / total > 0.4:
        base = "Afternoon Warrior ⚔️"
    else:
        base = "All-Day Coder 💻"
    
    # Add intensity modifier
    if peak_day_messages > 3000:
        return f"Hyperfocused {base}"
    elif total_sessions > 150:
        return f"Prolific {base}"
    else:
        return base


def generate_wrapped_stats(
    claude_dir: Optional[Path] = None,
    code_dir: Optional[Path] = None,
    include_hidden: bool = False,
    include_nested_repos: bool = False,
    max_file_size_mb: Optional[float] = None,
    relax_excludes: bool = False,
    include_git: bool = True,
    max_stats: bool = False,
) -> WrappedStats:
    """Generate all wrapped statistics from Claude Code data."""
    
    if claude_dir is None:
        claude_dir = Path.home() / ".claude"

    if code_dir is None:
        default_code_dir = Path.home() / "Claude Code Projects"
        code_dir = default_code_dir if default_code_dir.exists() else None

    if max_stats:
        include_hidden = True
        include_nested_repos = True
        relax_excludes = True
        if max_file_size_mb is None:
            max_file_size_mb = 10

    exclude_dirs = resolve_exclude_dirs(relax_excludes)
    if max_file_size_mb is None:
        max_file_size_bytes = DEFAULT_MAX_FILE_SIZE_BYTES
    else:
        max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
    
    # Parse raw data
    stats = parse_stats_cache(claude_dir)
    projects = get_projects(claude_dir)
    
    # Extract daily activity
    daily = stats.get('dailyActivity', [])
    
    # Calculate totals
    total_tool_calls = sum(d.get('toolCallCount', 0) for d in daily)
    
    # Find peak day
    peak_day = max(daily, key=lambda x: x.get('messageCount', 0)) if daily else {}
    
    # Get date range
    dates = [d['date'] for d in daily]
    date_range_start = min(dates) if dates else "N/A"
    date_range_end = max(dates) if dates else "N/A"
    
    # Model usage
    model_usage = stats.get('modelUsage', {})
    primary_model = "unknown"
    primary_output_tokens = 0
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation = 0

    for model, usage in model_usage.items():
        model_output = int(usage.get('outputTokens', 0) or 0)
        if model_output > primary_output_tokens:
            primary_output_tokens = model_output
            primary_model = model

        input_tokens += int(usage.get('inputTokens', 0) or 0)
        output_tokens += model_output
        cache_read += int(usage.get('cacheReadInputTokens', 0) or 0)
        cache_creation += int(usage.get('cacheCreationInputTokens', 0) or 0)
    
    # Hour distribution
    hour_counts = stats.get('hourCounts', {})
    peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0
    peak_hour_sessions = hour_counts.get(str(peak_hour), 0)
    
    # Longest session
    longest = stats.get('longestSession', {})
    duration_ms = longest.get('duration', 0)
    duration_hours = duration_ms / (1000 * 60 * 60)
    longest_date = longest.get('timestamp', '')[:10] if longest.get('timestamp') else 'N/A'
    
    # Calculate streak
    streak = calculate_streak(daily)
    
    # Determine personality
    personality = determine_personality(
        hour_counts, 
        stats.get('totalSessions', 0),
        peak_day.get('messageCount', 0)
    )
    
    # Format model name nicely
    model_display = primary_model.replace('claude-', '').replace('-20251101', '')
    # Convert opus-4-5 to Opus 4.5
    parts = model_display.split('-')
    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        model_display = f"{parts[0].title()} {parts[1]}.{parts[2]}"
    else:
        model_display = model_display.replace('-', ' ').title()

    model_percentage = (primary_output_tokens / output_tokens * 100.0) if output_tokens else 0.0
    total_tokens = input_tokens + output_tokens + cache_read + cache_creation
    
    repo_paths = find_git_repos(code_dir, include_hidden, include_nested_repos, exclude_dirs)
    codebase_stats = generate_codebase_stats(
        code_dir,
        include_hidden,
        max_file_size_bytes,
        exclude_dirs,
    )
    git_stats = (
        generate_git_stats(repo_paths)
        if include_git
        else {
            "repo_count": len(repo_paths),
            "commit_count": 0,
            "lines_added": 0,
            "lines_deleted": 0,
            "commit_hour_distribution": {str(i): 0 for i in range(24)},
            "peak_commit_hour": 0,
            "peak_commit_hour_commits": 0,
        }
    )

    return WrappedStats(
        total_sessions=stats.get('totalSessions', 0),
        total_messages=stats.get('totalMessages', 0),
        total_tool_calls=total_tool_calls,
        total_days_active=len(daily),
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        total_tokens=total_tokens,
        primary_model=model_display,
        model_percentage=round(model_percentage, 1),
        peak_day_date=peak_day.get('date', 'N/A'),
        peak_day_messages=peak_day.get('messageCount', 0),
        peak_day_tool_calls=peak_day.get('toolCallCount', 0),
        peak_day_sessions=peak_day.get('sessionCount', 0),
        peak_hour=int(peak_hour),
        peak_hour_sessions=peak_hour_sessions,
        hour_distribution=hour_counts,
        longest_session_duration_hours=round(duration_hours, 1),
        longest_session_messages=longest.get('messageCount', 0),
        longest_session_date=longest_date,
        projects=projects,
        project_count=len(projects),
        codebase_repo_count=codebase_stats["repo_count"],
        codebase_projects=codebase_stats["projects"],
        codebase_file_count=codebase_stats["file_count"],
        codebase_line_count=codebase_stats["line_count"],
        codebase_edit_hour_distribution=codebase_stats["edit_hour_distribution"],
        codebase_peak_edit_hour=codebase_stats["peak_edit_hour"],
        codebase_peak_edit_hour_files=codebase_stats["peak_edit_hour_files"],
        git_repo_count=git_stats["repo_count"],
        git_commit_count=git_stats["commit_count"],
        git_lines_added=git_stats["lines_added"],
        git_lines_deleted=git_stats["lines_deleted"],
        git_commit_hour_distribution=git_stats["commit_hour_distribution"],
        git_peak_commit_hour=git_stats["peak_commit_hour"],
        git_peak_commit_hour_commits=git_stats["peak_commit_hour_commits"],
        tokens_as_pages=int(output_tokens / 187),  # ~250 words/page, 0.75 tokens/word
        coding_personality=personality,
        streak_days=streak
    )


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Claude Code Wrapped stats')
    parser.add_argument('--claude-dir', type=Path, default=Path.home() / '.claude',
                       help='Path to .claude directory')
    parser.add_argument('--code-dir', type=Path, default=None,
                       help='Path to local code directory for repo stats')
    parser.add_argument('--include-hidden', action='store_true',
                       help='Include hidden directories in code scan')
    parser.add_argument('--include-nested-repos', action='store_true',
                       help='Include nested git repos in code scan')
    parser.add_argument('--max-file-size-mb', type=float, default=None,
                       help='Max file size (MB) to count for code stats')
    parser.add_argument('--relax-excludes', action='store_true',
                       help='Use a smaller exclude list for code scanning')
    parser.add_argument('--skip-git', action='store_true',
                       help='Skip git history stats')
    parser.add_argument('--max-stats', action='store_true',
                       help='Enable aggressive scanning for maximum stats')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON file path')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print JSON output')
    
    args = parser.parse_args()
    
    try:
        stats = generate_wrapped_stats(
            args.claude_dir,
            args.code_dir,
            include_hidden=args.include_hidden,
            include_nested_repos=args.include_nested_repos,
            max_file_size_mb=args.max_file_size_mb,
            relax_excludes=args.relax_excludes,
            include_git=not args.skip_git,
            max_stats=args.max_stats,
        )
        output = asdict(stats)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2 if args.pretty else None)
            print(f"Stats written to {args.output}")
        else:
            print(json.dumps(output, indent=2 if args.pretty else None))
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
