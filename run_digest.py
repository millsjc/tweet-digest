#!/usr/bin/env python3
import json
import hashlib
import mimetypes
import os
import sqlite3
import subprocess
from datetime import datetime
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

LIST_ID = os.environ.get("TWEET_DIGEST_LIST_ID", "1889690040562815368")
SENDER_EMAIL = os.environ.get("TWEET_DIGEST_SENDER_EMAIL", "mills2k4@hotmail.com")
KINDLE_EMAIL = os.environ.get("TWEET_DIGEST_KINDLE_EMAIL", "mills2k4_9aUC84@kindle.com")
SUMMARY_EMAIL = os.environ.get("TWEET_DIGEST_SUMMARY_EMAIL", SENDER_EMAIL)
ZOTERO_DIR = Path(os.environ.get("TWEET_DIGEST_ZOTERO_DIR", "/Users/camill/Zotero"))

SHORT_DOMAINS = {
    "x.com",
    "twitter.com",
    "t.co",
    "forms.gle",
    "docs.google.com",
    "calendar.google.com",
    "eventbrite.com",
    "meetup.com",
}
LONG_HINTS = [
    "blog",
    "arxiv.org",
    ".pdf",
    "paper",
    "report",
    "engineering",
    "research",
    "publication",
]


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def classify_url(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    path = parsed.path.lower()
    if domain in SHORT_DOMAINS:
        return "summary"
    if domain in {"gist.github.com", "gist.githubusercontent.com"}:
        return "kindle"
    if domain == "karpathy.ai" and "microgpt" in path:
        return "kindle"
    url_lower = url.lower()
    if any(hint in url_lower for hint in LONG_HINTS):
        return "kindle"
    return "summary"


def truncate(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_urls_from_legacy(legacy: dict) -> list[str]:
    urls = []
    for url_obj in legacy.get("entities", {}).get("urls", []):
        expanded = url_obj.get("expanded_url")
        if expanded:
            urls.append(expanded)
    return urls


def collect_urls_from_result(result: dict, collected_urls: list[str]) -> None:
    if not result:
        return
    if result.get("__typename") != "Tweet":
        return

    legacy = result.get("legacy", {})
    collected_urls.extend(extract_urls_from_legacy(legacy))

    quoted_result = result.get("quoted_status_result", {}).get("result")
    if quoted_result:
        collect_urls_from_result(quoted_result, collected_urls)

    retweet_result = legacy.get("retweeted_status_result", {}).get("result")
    if retweet_result:
        collect_urls_from_result(retweet_result, collected_urls)


def parse_timeline_tweets(data: list[dict]) -> list[dict]:
    tweets = []
    for entry in data:
        raw = entry.get("_raw", {})
        legacy = raw.get("legacy", {})
        user_results = raw.get("core", {}).get("user_results", {}).get("result", {})

        screen_name = (
            entry.get("author", {}).get("username")
            or user_results.get("legacy", {}).get("screen_name")
            or "unknown"
        )
        tweet_id = str(entry.get("id") or legacy.get("id_str") or "")
        text = legacy.get("full_text") or entry.get("text") or ""

        collected_urls = []
        collect_urls_from_result(raw, collected_urls)
        deduped_urls = []
        seen = set()
        for url in collected_urls:
            if url in seen:
                continue
            seen.add(url)
            deduped_urls.append(url)

        tweet_url = ""
        if screen_name != "unknown" and tweet_id:
            tweet_url = f"https://x.com/{screen_name}/status/{tweet_id}"

        tweets.append(
            {
                "screen_name": screen_name,
                "tweet_id": tweet_id,
                "tweet_url": tweet_url,
                "text": text,
                "links": deduped_urls,
            }
        )
    return tweets


def is_cjk_text(text: str) -> bool:
    for char in text:
        code = ord(char)
        if 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            return True
    return False


def translate_to_english(text: str) -> str:
    query = urlencode(
        {
            "client": "gtx",
            "sl": "auto",
            "tl": "en",
            "dt": "t",
            "q": text,
        }
    )
    request = Request(
        f"https://translate.googleapis.com/translate_a/single?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""

    if not isinstance(payload, list) or not payload:
        return ""
    chunks = payload[0]
    if not isinstance(chunks, list):
        return ""
    translated = "".join(chunk[0] for chunk in chunks if isinstance(chunk, list) and chunk and chunk[0])
    return translated.strip()


def clean_tweet_text(text: str) -> str:
    return " ".join(text.split())


def arxiv_abs_to_pdf(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "arxiv.org":
        return ""
    if not parsed.path.startswith("/abs/"):
        return ""
    paper_id = parsed.path.removeprefix("/abs/").strip("/")
    if not paper_id:
        return ""
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def write_eml(
    target: Path,
    subject: str,
    body: str,
    from_email: str,
    to_email: str = "",
    attachments: list[Path] | None = None,
) -> None:
    message = EmailMessage()
    if from_email:
        message["From"] = from_email
    if to_email:
        message["To"] = to_email
    message["Subject"] = subject
    message["X-Unsent"] = "1"
    message.set_content(body)

    for attachment in attachments or []:
        if not attachment.exists():
            continue
        mime_type, _ = mimetypes.guess_type(str(attachment))
        if not mime_type:
            mime_type = "application/octet-stream"
        main_type, sub_type = mime_type.split("/", 1)
        message.add_attachment(
            attachment.read_bytes(),
            maintype=main_type,
            subtype=sub_type,
            filename=attachment.name,
        )

    target.write_bytes(message.as_bytes(policy=SMTP))


def make_mailto_link(to_email: str, subject: str, body: str) -> str:
    to_part = quote(to_email) if to_email else ""
    subject_part = quote(subject)
    body_part = quote(body)
    return f"mailto:{to_part}?subject={subject_part}&body={body_part}"


def resolve_web2epub_dir(root_dir: Path) -> Path:
    explicit = os.environ.get("WEB2EPUB_DIR")
    if explicit:
        return Path(explicit)
    for candidate in (root_dir / "web2epub", root_dir / "web_to_epub"):
        if candidate.exists():
            return candidate
    return root_dir / "web_to_epub"


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def save_state(state_path: Path, payload: dict) -> None:
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tweet_id_from_url(tweet_url: str) -> str:
    parsed = urlparse(tweet_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 3 and parts[-2] == "status":
        return parts[-1]
    return ""


def tweet_identity(tweet: dict) -> str:
    tweet_id = (tweet.get("tweet_id") or "").strip()
    if tweet_id:
        return f"id:{tweet_id}"
    tweet_url = (tweet.get("tweet_url") or "").strip()
    if tweet_url:
        url_id = tweet_id_from_url(tweet_url)
        if url_id:
            return f"id:{url_id}"
        return f"url:{tweet_url}"
    seed = "|".join(
        [
            tweet.get("screen_name", ""),
            tweet.get("text", ""),
            *tweet.get("links", []),
        ]
    )
    return f"hash:{hashlib.sha1(seed.encode('utf-8')).hexdigest()}"


def seed_processed_items_from_links(links_json: Path) -> list[str]:
    if not links_json.exists():
        return []
    try:
        payload = json.loads(links_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []

    seeded: list[str] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        tweet_url = str(item.get("tweet_url") or "")
        tweet_id = tweet_id_from_url(tweet_url)
        key = f"id:{tweet_id}" if tweet_id else (f"url:{tweet_url}" if tweet_url else "")
        if not key or key in seen:
            continue
        seen.add(key)
        seeded.append(key)
    return seeded


def fetch_recent_zotero_pdfs(hours: int = 24) -> list[dict]:
    zotero_db = ZOTERO_DIR / "zotero.sqlite"
    zotero_storage = ZOTERO_DIR / "storage"
    if not zotero_db.exists() or not zotero_storage.exists():
        return []

    query = """
        SELECT i.key AS attachment_key, i.dateAdded AS added_utc, ia.path
        FROM itemAttachments ia
        JOIN items i ON i.itemID = ia.itemID
        WHERE ia.contentType = 'application/pdf'
          AND ia.path LIKE 'storage:%'
          AND i.dateAdded >= datetime('now', ?)
        ORDER BY i.dateAdded DESC
    """
    uri = f"file:{zotero_db}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return []

    try:
        rows = conn.execute(query, (f"-{hours} hours",)).fetchall()
    except sqlite3.Error:
        conn.close()
        return []
    conn.close()

    results: list[dict] = []
    seen: set[str] = set()
    for attachment_key, added_utc, path_value in rows:
        if not isinstance(path_value, str) or not path_value.startswith("storage:"):
            continue
        relative_part = path_value.removeprefix("storage:").replace("\\", "/").lstrip("/")
        candidate_path = zotero_storage / str(attachment_key) / relative_part
        if not candidate_path.exists():
            continue
        candidate_str = str(candidate_path.resolve())
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        results.append(
            {
                "attachment_key": str(attachment_key),
                "added_utc": str(added_utc),
                "path": candidate_str,
                "filename": candidate_path.name,
            }
        )
    return results


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    web2epub_dir = resolve_web2epub_dir(root_dir)
    draft_dir = root_dir / "email_drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)

    daily_raw = script_dir / "daily_raw.json"
    links_json = script_dir / "links.json"
    state_path = script_dir / "digest_state.json"
    kindle_urls_txt = script_dir / "kindle_urls.txt"
    pdf_dir = script_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    seeded_processed_items = seed_processed_items_from_links(links_json)

    with daily_raw.open("w", encoding="utf-8") as f:
        subprocess.run(
            ["bird", "list-timeline", LIST_ID, "--json-full"],
            stdout=f,
            check=True,
            text=True,
        )

    timeline_data = json.loads(daily_raw.read_text(encoding="utf-8"))
    tweets = parse_timeline_tweets(timeline_data)
    link_tweets = [tweet for tweet in tweets if tweet["links"]]

    state = load_state(state_path)
    processed_items = [item for item in state.get("processed_items", []) if isinstance(item, str)]
    processed_set = set(processed_items)
    if not processed_set and seeded_processed_items:
        processed_items.extend(seeded_processed_items)
        processed_set.update(seeded_processed_items)

    digest_tweets = []
    for tweet in link_tweets:
        key = tweet_identity(tweet)
        if key in processed_set:
            continue
        digest_tweets.append(tweet)

    extracted_items = []
    for tweet in digest_tweets:
        for url in tweet["links"]:
            extracted_items.append(
                {
                    "screen_name": tweet["screen_name"],
                    "text": tweet["text"],
                    "url": url,
                    "tweet_url": tweet["tweet_url"],
                }
            )
    links_json.write_text(json.dumps(extracted_items, indent=2) + "\n", encoding="utf-8")

    kindle_source_urls: list[str] = []
    kindle_source_set: set[str] = set()
    web_kindle_urls: list[str] = []
    web_kindle_set: set[str] = set()
    pdf_download_urls: list[str] = []
    pdf_download_set: set[str] = set()

    for tweet in digest_tweets:
        for url in tweet["links"]:
            pdf_url = ""
            if is_pdf_url(url):
                pdf_url = url
            else:
                arxiv_pdf = arxiv_abs_to_pdf(url)
                if arxiv_pdf:
                    pdf_url = arxiv_pdf

            if pdf_url:
                if url not in kindle_source_set:
                    kindle_source_set.add(url)
                    kindle_source_urls.append(url)
                if pdf_url not in pdf_download_set:
                    pdf_download_set.add(pdf_url)
                    pdf_download_urls.append(pdf_url)
                continue

            if classify_url(url) == "kindle":
                if url not in kindle_source_set:
                    kindle_source_set.add(url)
                    kindle_source_urls.append(url)
                if url not in web_kindle_set:
                    web_kindle_set.add(url)
                    web_kindle_urls.append(url)

    if kindle_source_urls:
        kindle_urls_txt.write_text("\n".join(kindle_source_urls) + "\n", encoding="utf-8")
    else:
        kindle_urls_txt.write_text("", encoding="utf-8")

    downloaded_pdfs = []
    for pdf_url in pdf_download_urls:
        name = Path(urlparse(pdf_url).path).name or "document.pdf"
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        target = pdf_dir / name
        try:
            run(["curl", "--fail", "-L", pdf_url, "-o", str(target)])
            downloaded_pdfs.append(target)
        except subprocess.CalledProcessError:
            pass

    zotero_recent_pdfs = fetch_recent_zotero_pdfs(hours=24)

    date_tag = datetime.now().strftime("%Y-%m-%d")
    today = datetime.now().strftime("%b %d, %Y")
    epub_path = script_dir / f"long-form-readings-{date_tag}.epub"
    legacy_epub_path = script_dir / "output.epub"
    if legacy_epub_path.exists():
        legacy_epub_path.unlink()
    if epub_path.exists():
        epub_path.unlink()

    if web_kindle_urls and web2epub_dir.exists():
        title = f"Long Form Readings - {date_tag}"
        run(
            [
                "uv",
                "run",
                "python",
                "web_to_epub.py",
                "--title",
                title,
                "-o",
                str(epub_path),
                *web_kindle_urls,
            ],
            cwd=web2epub_dir,
        )

    summary_lines = [f"Daily AI Updates - {today}", ""]
    if not digest_tweets:
        summary_lines.append("No new tweets today.")
        summary_lines.append("")
    for tweet in digest_tweets:
        cleaned_tweet = clean_tweet_text(tweet["text"])
        translated = ""
        if is_cjk_text(cleaned_tweet):
            translated = translate_to_english(cleaned_tweet)
        tweet_label = "Tweet" if len(cleaned_tweet) <= 260 else "Tweet excerpt"
        tweet_excerpt = truncate(cleaned_tweet, 260)
        summary_lines.append(f"@{tweet['screen_name']}")
        if translated:
            summary_lines.append(f"Tweet (EN): {truncate(translated, 260)}")
        summary_lines.append(f"{tweet_label}: {tweet_excerpt}")
        summary_lines.append("Links:")
        for url in tweet["links"]:
            suffix = " [SENT TO KINDLE ✅]" if url in kindle_source_set else ""
            summary_lines.append(f"- {url}{suffix}")
        if tweet["tweet_url"]:
            summary_lines.append(f"Tweet URL: {tweet['tweet_url']}")
        summary_lines.append("")

    summary_lines.append("Zotero PDFs added in last 24 hours:")
    if zotero_recent_pdfs:
        for pdf in zotero_recent_pdfs:
            summary_lines.append(f"- {pdf['filename']} ({pdf['added_utc']})")
    else:
        summary_lines.append("- none")
    summary_lines.append("")

    summary_draft = draft_dir / f"daily_summary_{date_tag}.txt"
    summary_body = "\n".join(summary_lines).rstrip() + "\n"
    summary_draft.write_text(summary_body, encoding="utf-8")

    kindle_lines = [
        f"From: {SENDER_EMAIL}",
        f"To: {KINDLE_EMAIL}",
        f"Subject: AI Digest - {date_tag}",
        "",
        "Attachments:",
    ]
    kindle_attachment_paths: list[Path] = []
    if epub_path.exists():
        kindle_attachment_paths.append(epub_path)
    kindle_attachment_paths.extend(downloaded_pdfs)
    kindle_attachment_paths.extend(Path(pdf["path"]) for pdf in zotero_recent_pdfs)
    for path in kindle_attachment_paths:
        kindle_lines.append(f"- {path}")
    if not kindle_attachment_paths:
        kindle_lines.append("- (no attachments created)")
    kindle_lines.extend(
        [
            "",
            "Body:",
            "Attached are today's long-form readings.",
        ]
    )

    kindle_draft = draft_dir / f"kindle_{date_tag}.txt"
    kindle_body = "\n".join(kindle_lines).rstrip() + "\n"
    kindle_draft.write_text(kindle_body, encoding="utf-8")

    summary_eml = draft_dir / f"daily_summary_{date_tag}.eml"
    summary_attachments = [summary_draft]
    write_eml(
        summary_eml,
        subject=f"Daily AI Updates - {date_tag}",
        body=summary_body,
        from_email=SENDER_EMAIL,
        to_email=SUMMARY_EMAIL,
        attachments=summary_attachments,
    )

    kindle_eml = draft_dir / f"kindle_{date_tag}.eml"
    kindle_attachments: list[Path] = []
    kindle_attachments.extend(kindle_attachment_paths)
    write_eml(
        kindle_eml,
        subject=f"AI Digest - {date_tag}",
        body="Attached are today's long-form readings.\n",
        from_email=SENDER_EMAIL,
        to_email=KINDLE_EMAIL,
        attachments=kindle_attachments,
    )

    summary_mailto_body = "Daily AI digest prepared. Open the .eml draft for the full message body."

    launcher = draft_dir / f"open_email_drafts_{date_tag}.html"
    launcher.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<body>",
                f'<p><a href="{make_mailto_link(SUMMARY_EMAIL, f"Daily AI Updates - {date_tag}", summary_mailto_body)}">Open Daily Summary via mailto</a></p>',
                f'<p><a href="{summary_eml.as_uri()}">Open Daily Summary .eml draft</a></p>',
                f'<p><a href="{kindle_eml.as_uri()}">Open Kindle .eml draft (with attachments)</a></p>',
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    command_launcher = draft_dir / f"open_email_drafts_{date_tag}.command"
    command_launcher.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                f'BASE="{draft_dir!s}"',
                f'D="{date_tag}"',
                'open -a "Mail" "$BASE/daily_summary_${D}.eml"',
                'open -a "Mail" "$BASE/kindle_${D}.eml"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    command_launcher.chmod(0o755)

    print(f"Wrote {summary_draft}")
    print(f"Wrote {kindle_draft}")
    print(f"Wrote {summary_eml}")
    print(f"Wrote {kindle_eml}")
    print(f"Wrote {launcher}")
    print(f"Wrote {command_launcher}")
    print(f'BASE="{draft_dir}"')
    print(f'D="{date_tag}"')
    print('open -a "Mail" "$BASE/daily_summary_${D}.eml"')
    print('open -a "Mail" "$BASE/kindle_${D}.eml"')

    for tweet in link_tweets:
        key = tweet_identity(tweet)
        if key in processed_set:
            continue
        processed_items.append(key)
        processed_set.add(key)
    state["processed_items"] = processed_items[-5000:]
    state["last_run_at"] = datetime.now().isoformat(timespec="seconds")
    state["last_new_tweet_count"] = len(digest_tweets)
    state["last_seen_link_tweet_count"] = len(link_tweets)
    save_state(state_path, state)

    if epub_path.exists():
        print(f"Created {epub_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
