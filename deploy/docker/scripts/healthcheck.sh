#!/bin/bash
# Docker HEALTHCHECK script for MIRA container
# Checks that the MIRA API is responding and healthy

set -e

# Check MIRA health endpoint
RESPONSE=$(curl -sf http://localhost:1993/v1/api/health 2>/dev/null || echo '{"data":{"status":"error"}}')

# Extract status
STATUS=$(echo "$RESPONSE" | jq -r '.data.status // "error"')

if [ "$STATUS" = "healthy" ]; then
    exit 0
else
    echo "Health check failed: status=$STATUS"
    exit 1
fi
