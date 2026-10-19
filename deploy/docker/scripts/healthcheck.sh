#!/bin/bash
# Docker HEALTHCHECK probe for the MIRA container.
#
# One GET against the app's own health route on the port it binds, and a pass
# only when the response envelope's `.data.status` reads "healthy". Any other
# value exits 1, and a dead socket exits 1 too: `curl -sf` fails on a refused
# connection and its fallback envelope carries status "error", so an absent
# server and a sick server report the same verdict. The probe reads the JSON
# field, never the HTTP status code, so a 200 carrying an unhealthy payload
# fails and the container is reported unhealthy instead of ready.

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
