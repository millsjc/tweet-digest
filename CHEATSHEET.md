# Tweet Digest Cheatsheet

Simplified workflow for creating a daily digest from the "AI" Twitter list.

## 1. Get Latest Tweets
Fetch the full timeline JSON using the `bird` CLI.
*Ensure `AUTH_TOKEN` and `CT0` environment variables are set.*

```bash
# Fetch timeline to a file
bird list-timeline 1889690040562815368 --json-full > daily_raw.json
```

## 2. Extract Links
Use the `extract-links` tool to parse the JSON and get a list of URLs/Tweets.

```bash
# Run the extractor
extract-links daily_raw.json > links.json
```

## 3. Filter & Review (Manual/LLM Step)
Review `links.json`.
- **Short updates/Threads**: Keep for the "Daily Summary" text key.
- **Long-form articles/blogs**: Copy URLs for the Kindle Digest.

## 4. Convert to EPUB (Kindle Digest)
Use the `web2epub` tool to bundle long-form articles into a single eBook.

```bash
# Navigate to web2epub repo
cd ~/projects/2025/web_to_epub

# Run conversion (replace URLs with actual links)
uv run python web_to_epub.py \
  --title "AI Digest - $(date +%Y-%m-%d)" \
  -o ~/projects/2026/tweet-digest/output.epub \
  https://example.com/article1 \
  https://example.com/article2
```

> **Note**: If a URL is a PDF, download it manually instead of using web2epub.

## 5. Delivery

### Send to Kindle
Email the `output.epub` (and any downloaded PDFs) to:
**`mills2k4_9aUC84@kindle.com`**

### Daily Summary
Compile the short updates/threads into a text summary.
Label the Kindle items as **[SENT TO KINDLE]** in the summary.

**Example Summary:**
```text
Daily AI Updates - Feb 10, 2026

[SENT TO KINDLE] DeepSeek R1 Paper - https://arxiv.org/abs/...
[SENT TO KINDLE] Anthropic's New Compiler - https://anthropic.com/...

- @karpathy: "Just released minGPT v2..." (Thread)
- @ylecun: "AI is not magic..."
```
