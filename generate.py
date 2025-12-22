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
        'output_tokens': stats.output_tokens,
        'primary_model': stats.primary_model,
        'peak_day_date': stats.peak_day_date,
        'peak_day_messages': stats.peak_day_messages,
        'peak_day_tool_calls': stats.peak_day_tool_calls,
        'coding_personality': stats.coding_personality,
        'hour_distribution': stats.hour_distribution,
        'longest_session_duration_hours': stats.longest_session_duration_hours,
        'longest_session_messages': stats.longest_session_messages,
        'streak_days': stats.streak_days,
        'projects': stats.projects,
        'tokens_as_pages': stats.tokens_as_pages,
    }
    return json.dumps(data, indent=12)


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


def html_to_png(html_path: Path, png_path: Path, width: int = 520, height: int = 1200) -> Path:
    """Convert HTML to PNG using available tools"""
    
    # Try playwright first (best quality)
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': width, 'height': height})
            page.goto(f'file://{html_path.absolute()}')
            
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
        """
    )
    
    parser.add_argument('--claude-dir', type=Path, default=Path.home() / '.claude',
                       help='Path to .claude directory (default: ~/.claude)')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output file path (default: output/wrapped.png)')
    parser.add_argument('--html-only', action='store_true',
                       help='Only generate HTML, skip PNG conversion')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only output stats as JSON, no image generation')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎁 Claude Code Wrapped Generator")
    print("=" * 40)
    
    # Generate stats
    print(f"\n📊 Parsing stats from {args.claude_dir}...")
    try:
        stats = generate_wrapped_stats(args.claude_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have Claude Code installed and have used it at least once.")
        sys.exit(1)
    
    # Print summary
    print(f"\n✓ Found {stats.total_sessions} sessions across {stats.total_days_active} days")
    print(f"  • {stats.total_messages:,} messages exchanged")
    print(f"  • {stats.output_tokens:,} tokens generated")
    print(f"  • {stats.project_count} projects touched")
    print(f"  • Peak day: {stats.peak_day_date} ({stats.peak_day_messages:,} messages)")
    print(f"  • Your vibe: {stats.coding_personality}")
    
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
