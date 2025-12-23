#!/usr/bin/env python3
"""
Claude Code Wrapped - Image Generator
Generates a shareable wrapped image from your Claude Code stats
"""

import json
import sys
import subprocess
from pathlib import Path
from string import Template
import argparse
import shutil
from typing import Optional

from parse_stats import generate_wrapped_stats, WrappedStats

TEMPLATE_PATH = Path(__file__).parent / 'wrapped.html'
OUTPUT_DIR = Path(__file__).parent / 'output'


def format_number(num: int) -> str:
    """Format large numbers for display"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def stats_to_js_object(stats: WrappedStats) -> str:
    """Convert stats to JavaScript object literal"""
    data = {
        'total_sessions': stats.total_sessions,
        'total_messages': stats.total_messages,
        'total_tool_calls': stats.total_tool_calls,
        'total_days_active': stats.total_days_active,
        'date_range_start': stats.date_range_start,
        'date_range_end': stats.date_range_end,
        'input_tokens': stats.input_tokens,
        'output_tokens': stats.output_tokens,
        'cache_read_tokens': stats.cache_read_tokens,
        'cache_creation_tokens': stats.cache_creation_tokens,
        'total_tokens': stats.total_tokens,
        'estimated_total_tokens_conservative': stats.estimated_total_tokens_conservative,
        'estimated_tokens_basis_conservative': stats.estimated_tokens_basis_conservative,
        'primary_model': stats.primary_model,
        'peak_day_date': stats.peak_day_date,
        'peak_day_messages': stats.peak_day_messages,
        'peak_day_tool_calls': stats.peak_day_tool_calls,
        'coding_personality': stats.coding_personality,
        'hour_distribution': stats.hour_distribution,
        'peak_hour': stats.peak_hour,
        'peak_hour_sessions': stats.peak_hour_sessions,
        'session_day_distribution': stats.session_day_distribution,
        'session_peak_day': stats.session_peak_day,
        'session_peak_day_sessions': stats.session_peak_day_sessions,
        'longest_session_duration_hours': stats.longest_session_duration_hours,
        'longest_session_messages': stats.longest_session_messages,
        'streak_days': stats.streak_days,
        'projects': stats.projects,
        'tokens_as_pages': stats.tokens_as_pages,
        'codebase_repo_count': stats.codebase_repo_count,
        'codebase_projects': stats.codebase_projects,
        'codebase_file_count': stats.codebase_file_count,
        'codebase_line_count': stats.codebase_line_count,
        'codebase_edit_hour_distribution': stats.codebase_edit_hour_distribution,
        'codebase_peak_edit_hour': stats.codebase_peak_edit_hour,
        'codebase_peak_edit_hour_files': stats.codebase_peak_edit_hour_files,
        'git_repo_count': stats.git_repo_count,
        'git_commit_count': stats.git_commit_count,
        'git_lines_added': stats.git_lines_added,
        'git_lines_deleted': stats.git_lines_deleted,
        'git_commit_hour_distribution': stats.git_commit_hour_distribution,
        'git_peak_commit_hour': stats.git_peak_commit_hour,
        'git_peak_commit_hour_commits': stats.git_peak_commit_hour_commits,
        'git_commit_day_distribution': stats.git_commit_day_distribution,
        'git_peak_commit_day': stats.git_peak_commit_day,
        'git_peak_commit_day_commits': stats.git_peak_commit_day_commits,
        'git_commit_month_distribution': stats.git_commit_month_distribution,
        'git_peak_commit_month_label': stats.git_peak_commit_month_label,
        'git_peak_commit_month_commits': stats.git_peak_commit_month_commits,
        'git_longest_commit_streak_days': stats.git_longest_commit_streak_days,
        'git_longest_commit_gap_days': stats.git_longest_commit_gap_days,
        'git_commit_churn_hour_distribution': stats.git_commit_churn_hour_distribution,
        'git_peak_churn_hour': stats.git_peak_churn_hour,
        'git_peak_churn_hour_lines': stats.git_peak_churn_hour_lines,
        'git_repo_churn_recent': stats.git_repo_churn_recent,
        'language_mix': stats.language_mix,
        'github_available': stats.github_available,
        'github_year': stats.github_year,
        'github_total_contributions': stats.github_total_contributions,
        'github_total_commits': stats.github_total_commits,
        'github_total_prs': stats.github_total_prs,
        'github_total_reviews': stats.github_total_reviews,
        'github_total_issues': stats.github_total_issues,
        'github_total_repos': stats.github_total_repos,
        'github_owned_repo_count': stats.github_owned_repo_count,
        'github_created_repo_count': stats.github_created_repo_count,
        'github_created_repos': stats.github_created_repos,
        'github_day_distribution': stats.github_day_distribution,
        'github_peak_day': stats.github_peak_day,
        'github_peak_day_contributions': stats.github_peak_day_contributions,
    }
    return json.dumps(data, indent=12)


def _prompt_path(prompt: str) -> Optional[Path]:
    if not sys.stdin.isatty():
        return None
    try:
        value = input(prompt).strip()
    except EOFError:
        return None
    if not value:
        return None
    return Path(value).expanduser()


def _gh_is_authenticated() -> bool:
    if shutil.which("gh") is None:
        return False
    try:
        subprocess.run(["gh", "auth", "status"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        return False
    return True


def generate_html(stats: WrappedStats, output_path: Path) -> Path:
    """Generate HTML file with embedded stats"""
    
    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()
    
    # Replace the WRAPPED_DATA object in the template
    js_data = stats_to_js_object(stats)
    
    # Find and replace the WRAPPED_DATA constant
    import re
    pattern = r'const WRAPPED_DATA = \{[^}]+\};'
    replacement = f'const WRAPPED_DATA = {js_data};'
    
    # More robust replacement - find the entire WRAPPED_DATA block
    start_marker = 'const WRAPPED_DATA = {'
    end_marker = '};'
    
    start_idx = template.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find WRAPPED_DATA in template")
    
    # Find the matching closing brace
    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(template[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 2  # Include the };
                break
    
    html = template[:start_idx] + f'const WRAPPED_DATA = {js_data};' + template[end_idx:]
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path


def html_to_png(html_path: Path, png_path: Path, width: int = 2000, height: int = 1200) -> Path:
    """Convert HTML to PNG using available tools"""
    
    # Try playwright first (best quality)
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': width, 'height': height})
            page.goto(f'file://{html_path.absolute()}', wait_until='domcontentloaded', timeout=60000)
            
            # Wait for fonts to load
            page.wait_for_timeout(1000)
            
            # Screenshot just the card
            card = page.locator('#wrapped-card')
            card.screenshot(path=str(png_path))
            
            browser.close()
        
        print(f"✓ Generated PNG with Playwright: {png_path}")
        return png_path
        
    except ImportError:
        pass
    
    # Try selenium as fallback
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument(f'--window-size={width},{height}')
        
        driver = webdriver.Chrome(options=options)
        driver.get(f'file://{html_path.absolute()}')
        
        import time
        time.sleep(2)  # Wait for fonts
        
        card = driver.find_element('id', 'wrapped-card')
        card.screenshot(str(png_path))
        
        driver.quit()
        
        print(f"✓ Generated PNG with Selenium: {png_path}")
        return png_path
        
    except ImportError:
        pass
    
    # No screenshot tool available
    print("⚠ No screenshot tool available (playwright or selenium)")
    print(f"  HTML file generated at: {html_path}")
    print("  Open it in a browser and take a screenshot manually")
    print("\n  To enable automatic screenshots, install playwright:")
    print("    pip install playwright")
    print("    playwright install chromium")
    
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate your Claude Code Wrapped image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py                    # Generate from ~/.claude
  python generate.py --output my_wrapped.png
  python generate.py --html-only        # Just generate HTML
  python generate.py --stats-only       # Just print stats as JSON
  python generate.py --code-dir ~/Projects
  python generate.py --max-stats
        """
    )
    
    parser.add_argument('--claude-dir', type=Path, default=Path.home() / '.claude',
                       help='Path to .claude directory (default: ~/.claude)')
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
    parser.add_argument('--redact-projects', action='store_true',
                       help='Redact project names in output')
    parser.add_argument('--redact-prefix-len', type=int, default=3,
                       help='Prefix length used for redacted project names')
    parser.add_argument('--estimate-tokens', action='store_true',
                       help='Estimate total tokens across all projects')
    parser.add_argument('--estimate-by-commits', action='store_true',
                       help='Scale token estimate using git commit history')
    parser.add_argument('--github-stats', action='store_true',
                       help='Include GitHub contributions via gh CLI')
    parser.add_argument('--github-year', type=int, default=None,
                       help='Year to use for GitHub contributions')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output file path (default: output/wrapped.png)')
    parser.add_argument('--html-only', action='store_true',
                       help='Only generate HTML, skip PNG conversion')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only output stats as JSON, no image generation')
    
    args = parser.parse_args()

    if args.code_dir is None:
        prompt = "Project folder to scan for repos (leave blank to skip): "
        selected = _prompt_path(prompt)
        if selected:
            args.code_dir = selected

    if not args.github_stats and sys.stdin.isatty():
        try:
            include = input("Include GitHub stats via gh CLI? [y/N]: ").strip().lower()
        except EOFError:
            include = ""
        if include in {"y", "yes"}:
            args.github_stats = True

    if args.github_stats:
        if shutil.which("gh") is None:
            print("⚠ gh CLI not found; skipping GitHub stats.")
            args.github_stats = False
        elif not _gh_is_authenticated():
            if sys.stdin.isatty():
                try:
                    login = input("gh CLI not logged in. Run `gh auth login` now? [y/N]: ").strip().lower()
                except EOFError:
                    login = ""
                if login in {"y", "yes"}:
                    subprocess.run(["gh", "auth", "login"])
            if not _gh_is_authenticated():
                print("⚠ gh CLI not authenticated; skipping GitHub stats.")
                args.github_stats = False
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎁 Claude Code Wrapped Generator")
    print("=" * 40)
    
    # Generate stats
    print(f"\n📊 Parsing stats from {args.claude_dir}...")
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
            redact_projects=args.redact_projects,
            redact_prefix_len=args.redact_prefix_len,
            estimate_tokens=args.estimate_tokens,
            estimate_by_commits=args.estimate_by_commits,
            include_github=args.github_stats,
            github_year=args.github_year,
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have Claude Code installed and have used it at least once.")
        sys.exit(1)
    
    # Print summary
    print(f"\n✓ Found {stats.total_sessions} sessions across {stats.total_days_active} days")
    print(f"  • {stats.total_messages:,} messages exchanged")
    print(f"  • {stats.total_tokens:,} tokens processed (incl. cache)")
    print(f"  • {stats.output_tokens:,} output tokens")
    if stats.estimated_total_tokens_conservative:
        cons = stats.estimated_total_tokens_conservative
        basis = stats.estimated_tokens_basis_conservative or "estimate"
        print(f"  • Estimated tokens (conservative): {cons:,} ({basis})")
    print(f"  • {stats.project_count} projects touched")
    print(f"  • Peak day: {stats.peak_day_date} ({stats.peak_day_messages:,} messages)")
    print(f"  • Your vibe: {stats.coding_personality}")
    if stats.github_available:
        print(f"  • GitHub {stats.github_year}: {stats.github_total_contributions:,} contributions")
    
    if args.stats_only:
        print("\n" + "=" * 40)
        print(json.dumps(stats.__dict__, indent=2, default=list))
        return
    
    # Generate HTML
    html_path = OUTPUT_DIR / 'wrapped.html'
    print(f"\n🎨 Generating HTML...")
    generate_html(stats, html_path)
    print(f"  ✓ {html_path}")
    
    if args.html_only:
        print(f"\n✅ Done! Open {html_path} in your browser")
        return
    
    # Generate PNG
    png_path = args.output or (OUTPUT_DIR / 'wrapped.png')
    print(f"\n📸 Generating PNG...")
    result = html_to_png(html_path, png_path)
    
    if result.suffix == '.png':
        print(f"\n✅ Your Claude Code Wrapped is ready!")
        print(f"   {png_path}")
        print("\n   Share it on socials! 🚀")
    

if __name__ == '__main__':
    main()
