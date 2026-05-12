#!/usr/bin/env bash
# Fetch a Jira issue via REST API and output structured JSON.
# Usage: bash fetch_issue.sh <ISSUE_KEY>
# Requires: JIRA_EMAIL, JIRA_API_TOKEN environment variables.

set -euo pipefail

JIRA_BASE_URL="https://rnd-hub.atlassian.net"

ISSUE_KEY="${1:?Usage: fetch_issue.sh <ISSUE_KEY>}"

if [[ -z "${JIRA_EMAIL:-}" ]]; then
  echo "ERROR: JIRA_EMAIL environment variable is not set." >&2
  echo "Set it to your Atlassian account email." >&2
  exit 1
fi

if [[ -z "${JIRA_API_TOKEN:-}" ]]; then
  echo "ERROR: JIRA_API_TOKEN environment variable is not set." >&2
  echo "Create one at: https://id.atlassian.com/manage-profile/security/api-tokens" >&2
  exit 1
fi

FIELDS="summary,status,issuetype,priority,assignee,description,labels,components,subtasks,comment"

API_URL="${JIRA_BASE_URL}/rest/api/3/issue/${ISSUE_KEY}?fields=${FIELDS}"

curl -sS --fail-with-body \
  -u "${JIRA_EMAIL}:${JIRA_API_TOKEN}" \
  -H "Accept: application/json" \
  "${API_URL}" \
  | python3 -m json.tool
