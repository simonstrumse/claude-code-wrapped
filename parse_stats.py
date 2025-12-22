#!/usr/bin/env python3
"""
Claude Code Wrapped - Stats Parser
Parses ~/.claude/ data to generate wrapped statistics
"""

import json
import os
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


def get_projects(claude_dir: Path) -> list[str]:
    """Get list of projects from projects directory"""
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return []
    
    projects = []
    for item in projects_dir.iterdir():
        if item.is_dir():
            # Folder names are like: -Users-franciscoleal-Documents-workspace-talentprotocol-talent-api
            # We want just the last part: talent-api
            parts = item.name.split('-')
            
            # Find the last meaningful segment (after workspace/personal/etc)
            # Look for common markers and take everything after
            name_parts = []
            capture = False
            
            for i, part in enumerate(parts):
                if part in ['workspace', 'projects', 'repos', 'code', 'dev']:
                    capture = True
                    continue
                if capture and part not in ['personal', 'work', 'talentprotocol', 'Documents']:
                    name_parts.append(part)
            
            if name_parts:
                project_name = '-'.join(name_parts)
            else:
                # Fallback: just take the last non-empty part
                project_name = [p for p in parts if p][-1] if parts else item.name
            
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


def generate_wrapped_stats(claude_dir: Optional[Path] = None) -> WrappedStats:
    """Generate all wrapped statistics from Claude Code data"""
    
    if claude_dir is None:
        claude_dir = Path.home() / ".claude"
    
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
    primary_model = list(model_usage.keys())[0] if model_usage else "unknown"
    
    # Token stats from primary model
    primary_stats = model_usage.get(primary_model, {})
    input_tokens = primary_stats.get('inputTokens', 0)
    output_tokens = primary_stats.get('outputTokens', 0)
    cache_read = primary_stats.get('cacheReadInputTokens', 0)
    cache_creation = primary_stats.get('cacheCreationInputTokens', 0)
    
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
        primary_model=model_display,
        model_percentage=100.0,  # Can calculate if multiple models
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
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON file path')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print JSON output')
    
    args = parser.parse_args()
    
    try:
        stats = generate_wrapped_stats(args.claude_dir)
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
