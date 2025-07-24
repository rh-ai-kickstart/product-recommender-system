#!/bin/bash

# Fetch the latest main branch from origin
git fetch origin main

# Count how many commits are in the current branch that are not in origin/main
COMMITS_AHEAD=$(git rev-list --count upstream/main..HEAD)

# Check if there are more than 5 commits
if [ "$COMMITS_AHEAD" -gt 5 ]; then
    echo "PR contains too many commits: $COMMITS_AHEAD commits ahead of origin/main"
    exit 1
else
    echo "PR has $COMMITS_AHEAD commits ahead of origin/main, which is within the limit"
    exit 0
fi
