# Tweet Digest Automation

## Goal
Automate the discovery and delivery of high-quality reading material from Twitter/X lists (specifically the "AI" list) to Kindle.

The pipeline should:
1.  **Fetch**: Retrieve tweets from the "AI" list using `bird`.
2.  **Extract**: Parse tweets to find links to external content (blogs, PDFs, reports).
3.  **Filter/Classify**: Use an LLM to decide:
    *   Is this "reading material" (long-form, article, paper)? -> **Send to Kindle**.
    *   Is this just a short update/announcement? -> **Include in Daily Summary Email**.
4.  **Convert**: Use `Web2Epub` to convert readable URLs to EPUB format.
5.  **Deliver**:
    *   Email EPUBs to the user's Kindle email address.
    *   Email a daily summary of all interesting tweets to the user's standard email.

## Test Data (Derived from Initial "AI" List Fetch)

The following links were extracted from a sample run of the `bird` CLI on the "AI" list (Feb 2026):

*   **Building a C Compiler (Anthropic Engineering Blog)**
    *   Source: @OwainEvans_UK (RT @AnthropicAI)
    *   Link: https://www.anthropic.com/engineering/building-c-compiler
    *   *Action: Candidate for Kindle*

*   **International Scientific Report on the Safety of Advanced AI 2026**
    *   Source: @geoffreyhinton (RT @Yoshua_Bengio)
    *   Link: https://www.gov.uk/government/publications/international-scientific-report-on-the-safety-of-advanced-ai-2026
    *   *Action: Candidate for Kindle (PDF or Webpage)*

*   **AI and the Future of Work (AI Security Institute)**
    *   Source: @AISecurityInst
    *   Link: https://www.aisi.gov.uk/blog/ai-and-the-future-of-work-measuring-ai-driven-productivity-gains-for-workplace-tasks
    *   *Action: Candidate for Kindle*

*   **Sakana AI Beta Tester Recruitment Form**
    *   Source: @YesThisIsLion
    *   Link: https://forms.gle/R6KqqSTvUpXGEyHw5
    *   *Action: Daily Summary Only (Not reading material)*

## Architecture

1.  **Ingestion**: `bird list-timeline <list_id> --json-full`
2.  **Processing**: Python script (`extract_links.py`) to parse JSON and output structured list of items.
3.  **Intelligence**: Python/LLM script to analyze content of links (or text of tweet) and classify.
4.  **Conversion**: `Web2Epub` (CLI tool) for `URL -> EPUB`.
5.  **Notification**: SMTP/API to send emails.
