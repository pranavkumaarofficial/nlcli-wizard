#!/bin/bash
# docker-wizard.sh - Natural language wrapper for Docker CLI
#
# Installation (add to ~/.bashrc or ~/.zshrc):
#
#   # Option 1: Alias for standalone command
#   alias docker-w='bash /path/to/nlcli-wizard/scripts/docker-wizard.sh'
#
#   # Option 2: Override docker to support -w flag (all other args pass through)
#   docker() {
#       if [ "$1" = "-w" ]; then
#           shift
#           python -m nlcli_wizard.cli translate --cli-tool docker "$@"
#       else
#           command docker "$@"
#       fi
#   }

if [ -z "$1" ]; then
    echo "Usage: docker-wizard \"natural language query\""
    echo ""
    echo "Examples:"
    echo "  docker-wizard \"show all running containers\""
    echo "  docker-wizard \"run nginx on port 8080 in background\""
    echo "  docker-wizard \"build image tagged myapp version 2\""
    exit 1
fi

python -m nlcli_wizard.cli translate --cli-tool docker "$@"
