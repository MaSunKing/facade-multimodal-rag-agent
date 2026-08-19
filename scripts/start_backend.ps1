[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & $Python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --env-file $EnvFile
} finally {
    Pop-Location
}
