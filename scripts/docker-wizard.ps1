# docker-wizard.ps1 - Natural language wrapper for Docker CLI
#
# Installation (add to PowerShell profile):
#
#   # Option 1: Alias for standalone command
#   function docker-w { & python -m nlcli_wizard.cli translate --cli-tool docker @args }
#
#   # Option 2: Override docker to support -w flag
#   function docker {
#       if ($args[0] -eq "-w") {
#           $query = $args[1..($args.Length-1)]
#           & python -m nlcli_wizard.cli translate --cli-tool docker @query
#       } else {
#           & docker.exe @args
#       }
#   }

param(
    [Parameter(Mandatory=$true, Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$Query
)

$queryStr = $Query -join " "

if (-not $queryStr) {
    Write-Host "Usage: docker-wizard `"natural language query`""
    Write-Host ""
    Write-Host "Examples:"
    Write-Host '  docker-wizard "show all running containers"'
    Write-Host '  docker-wizard "run nginx on port 8080 in background"'
    Write-Host '  docker-wizard "build image tagged myapp version 2"'
    exit 1
}

& python -m nlcli_wizard.cli translate --cli-tool docker $queryStr
