#!/usr/bin/env python3
"""
Extract Links Tool
Usage: uv run extract_links.py <input_json_file>

Parses a Bird CLI JSON export (from 'bird list-timeline ... --json-full')
and outputs a JSON list of items containing:
- screen_name
- text
- url (expanded)
"""

import json
import sys
from typing import List, Dict, Any

def extract_urls_from_legacy(legacy: Dict[str, Any]) -> List[str]:
    urls = []
    if 'entities' in legacy and 'urls' in legacy['entities']:
        for u in legacy['entities']['urls']:
            expanded = u.get('expanded_url')
            if expanded:
                # Filter out pure twitter.com links that just repeat the status
                # (unless specifically desired, but usually redundant for a digest)
                # We keep them if they might be thread unrolls, but for now just pass through
                urls.append(expanded)
    return urls

def process_result_level(result: Dict[str, Any], collected_urls: List[str]):
    if not result: return 
    
    # Check typename
    typename = result.get('__typename')
    if typename == 'Tweet':
        legacy = result.get('legacy', {})
        collected_urls.extend(extract_urls_from_legacy(legacy))
        
        # Check quoted status
        if 'quoted_status_result' in result:
             q_res = result['quoted_status_result'].get('result')
             process_result_level(q_res, collected_urls)
             
        # Check retweeted status
        if 'retweeted_status_result' in legacy:
             rt_res = legacy['retweeted_status_result'].get('result')
             process_result_level(rt_res, collected_urls)

def extract_from_file(filepath: str) -> List[Dict[str, Any]]:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        return []

    extracted_items = []

    for tweet in data:
        _raw = tweet.get('_raw')
        if not _raw: continue
        
        # Main Tweet Info
        legacy = _raw.get('legacy', {})
        user_results = _raw.get('core', {}).get('user_results', {}).get('result', {})
        screen_name = user_results.get('legacy', {}).get('screen_name', 'unknown')
        full_text = legacy.get('full_text', '')
        
        # recursive extraction
        collected_urls = []
        process_result_level(_raw, collected_urls)

        # Dedup locally for this tweet
        seen = set()
        for u in collected_urls:
             if u not in seen:
                 seen.add(u)
                 extracted_items.append({
                     'screen_name': screen_name,
                     'text': full_text,
                     'url': u
                 })
    
    return extracted_items

def main():
    if len(sys.argv) < 2:
        print("Usage: extract-links <input_json_file>", file=sys.stderr)
        sys.exit(1)
        
    filepath = sys.argv[1]
    items = extract_from_file(filepath)
    print(json.dumps(items, indent=2))

if __name__ == '__main__':
    main()
