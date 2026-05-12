---
name: jira-read-task
description: "Read a Jira task and display its requirements. Use when the user asks to read, fetch, view, or go to a Jira ticket, issue, task, story, or bug. Triggered by Jira URLs or issue keys like NSOC-12345."
argument-hint: "Jira issue key (e.g. NSOC-15849) or full URL (e.g. https://rnd-hub.atlassian.net/browse/NSOC-15849)"
---

# Read Jira Task

Fetch a Jira issue and present its requirements so the agent can act on them.

## Prerequisites

The following environment variables must be set (e.g. in a `.env` file or shell profile):

| Variable | Description |
|---|---|
| `JIRA_EMAIL` | Atlassian account email |
| `JIRA_API_TOKEN` | Atlassian API token ([create one](https://id.atlassian.com/manage-profile/security/api-tokens)) |

## Procedure

### 1. Extract the Issue Key

From the user's input, extract the Jira issue key:
- If a URL like `https://rnd-hub.atlassian.net/browse/NSOC-15849` → extract `NSOC-15849`
- If already a key like `NSOC-15849` → use directly
- The key format is `PROJECT-NUMBER` (uppercase letters, dash, digits)

### 2. Fetch the Issue

Run the fetch script to retrieve the issue:

```bash
bash .github/skills/jira-read-task/scripts/fetch_issue.sh <ISSUE_KEY>
```

The script uses `JIRA_EMAIL` and `JIRA_API_TOKEN` from the environment.

If the env vars are not set, tell the user to set them and provide the link to create an API token.

### 3. Present the Results

After fetching, present a structured summary:

- **Key**: issue key
- **Summary**: title
- **Status**: current status
- **Type**: story / bug / task / etc.
- **Priority**: priority level
- **Assignee**: who it's assigned to
- **Description**: full description (requirements)
- **Acceptance Criteria**: if present in the description
- **Subtasks**: list if any
- **Labels / Components**: if present

### 4. Offer Next Steps

After presenting, ask the user what they'd like to do:
- Start implementing the requirements
- Propose an OpenSpec change from the requirements
- Explore / ask questions about the requirements
