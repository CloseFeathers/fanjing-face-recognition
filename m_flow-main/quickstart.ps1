# ============================================================================
# M-Flow Quickstart — one command to launch the full stack (Windows)
# Usage:  .\quickstart.ps1
# ============================================================================
$ErrorActionPreference = "Stop"

function Write-Ok    { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }
function Exit-Fatal  { param($msg) Write-Err $msg; exit 1 }

function Show-Banner {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║                                          ║" -ForegroundColor Cyan
    Write-Host "  ║   M - F L O W   Q U I C K S T A R T     ║" -ForegroundColor Cyan
    Write-Host "  ║                                          ║" -ForegroundColor Cyan
    Write-Host "  ╚══════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Show-ReadyBanner {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "  ║  M-Flow is ready!                        ║" -ForegroundColor Green
    Write-Host "  ║                                          ║" -ForegroundColor Green
    Write-Host "  ║  Frontend : http://localhost:3000         ║" -ForegroundColor Green
    Write-Host "  ║  API      : http://localhost:8000         ║" -ForegroundColor Green
    Write-Host "  ║  API Docs : http://localhost:8000/docs    ║" -ForegroundColor Green
    Write-Host "  ║                                          ║" -ForegroundColor Green
    Write-Host "  ║  Press Ctrl+C to stop                    ║" -ForegroundColor Green
    Write-Host "  ╚══════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
}

# ── Safety ─────────────────────────────────────────────────────────────────

if (-not (Test-Path "pyproject.toml")) {
    Exit-Fatal "Not in M-Flow project root. Run this from the cloned repo directory."
}

# ── Environment checks ────────────────────────────────────────────────────

Show-Banner
Write-Host "  Checking environment..."

$hasDocker = $false
$hasPython = $false

try {
    $dockerVer = & docker --version 2>$null
    $composeVer = & docker compose version 2>$null
    if ($dockerVer -and $composeVer) {
        Write-Ok "Docker: $dockerVer"
        Write-Ok "Compose: $composeVer"
        $info = & docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Docker daemon running"
            $hasDocker = $true
        } else {
            Write-Warn "Docker installed but daemon not running"
        }
    }
} catch {
    Write-Warn "Docker not found (install: https://docs.docker.com/get-docker/)"
}

try {
    $pyVer = & python --version 2>$null
    if ($pyVer -match "3\.(\d+)") {
        $minor = [int]$Matches[1]
        if ($minor -ge 10) {
            Write-Ok $pyVer
            $hasPython = $true
        } else {
            Write-Warn "$pyVer (need 3.10+)"
        }
    }
} catch {
    Write-Warn "Python not found"
}

if (-not $hasDocker -and -not $hasPython) {
    Exit-Fatal "Neither Docker nor Python 3.10+ found."
}

# Port checks
foreach ($port in @(8000, 3000)) {
    try {
        $conn = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
        if ($conn.TcpTestSucceeded) {
            Write-Warn "Port $port already in use"
        } else {
            Write-Ok "Port $port available"
        }
    } catch {
        Write-Ok "Port $port available"
    }
}

# ── Mode selection ────────────────────────────────────────────────────────

Write-Host ""
Write-Host "  Select deployment mode:"
Write-Host ""
if ($hasDocker) {
    Write-Host "    [1] Docker (recommended) — Backend + Frontend"
    Write-Host "    [2] Docker + Neo4j"
    Write-Host "    [3] Docker + PostgreSQL"
}
if ($hasPython) {
    Write-Host "    [4] Local Python (no Docker)"
}
if ($hasDocker) {
    Write-Host "    [5] Custom Docker profiles"
}
Write-Host ""

$choice = Read-Host "  >"
if (-not $choice) { $choice = "1" }

# ── API Key configuration ────────────────────────────────────────────────

Write-Host ""
Write-Host "  Configuring environment..."

if (Test-Path ".env") {
    $ow = Read-Host "  .env already exists. Overwrite? [y/N]"
    if ($ow -eq "y" -or $ow -eq "Y") {
        Copy-Item ".env.template" ".env" -Force
        Write-Ok "Created fresh .env"
    } else {
        Write-Ok "Keeping existing .env"
    }
} else {
    Copy-Item ".env.template" ".env"
    Write-Ok "Created .env from template"
}

$currentKey = (Get-Content .env | Select-String "^LLM_API_KEY=" | ForEach-Object { $_ -replace 'LLM_API_KEY=|"', '' }).Trim()
if ($currentKey -eq "your_api_key" -or [string]::IsNullOrEmpty($currentKey)) {
    $secKey = Read-Host "  Enter your LLM API key" -AsSecureString
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secKey))
    if ($apiKey) {
        (Get-Content .env) -replace '^LLM_API_KEY=.*', "LLM_API_KEY=`"$apiKey`"" | Set-Content .env
        Write-Ok "API key saved"
    } else {
        Write-Warn "No key entered — set LLM_API_KEY in .env later"
    }
} else {
    Write-Ok "API key already configured"
}

$provider = Read-Host "  LLM provider [openai]"
if (-not $provider) { $provider = "openai" }
(Get-Content .env) -replace '^LLM_PROVIDER=.*', "LLM_PROVIDER=`"$provider`"" | Set-Content .env

$model = Read-Host "  LLM model [gpt-5-nano]"
if (-not $model) { $model = "gpt-5-nano" }
(Get-Content .env) -replace '^LLM_MODEL=.*', "LLM_MODEL=`"$provider/$model`"" | Set-Content .env

Write-Ok "Configuration saved to .env"

# ── Health check ─────────────────────────────────────────────────────────

function Wait-ForHealth {
    param($url, $timeout)
    $elapsed = 0
    while ($elapsed -lt $timeout) {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 3
        $elapsed += 3
        Write-Host "." -NoNewline
    }
    Write-Host ""
    return $false
}

# ── Launch ───────────────────────────────────────────────────────────────

function Launch-Docker {
    param($profiles)
    Write-Host ""
    Write-Host "  Starting M-Flow (first run may take 5-10 minutes)..."

    $cmd = "docker compose $profiles up -d --build"
    Invoke-Expression $cmd

    Write-Host "  Waiting for backend" -NoNewline
    if (Wait-ForHealth "http://localhost:8000/health" 300) { Write-Ok "Backend ready" }
    else { Write-Warn "Backend may still be starting" }

    Write-Host "  Waiting for frontend" -NoNewline
    if (Wait-ForHealth "http://localhost:3000" 120) { Write-Ok "Frontend ready" }
    else { Write-Warn "Frontend may still be building" }

    Show-ReadyBanner
    Start-Process "http://localhost:3000"

    Write-Host "  Streaming logs (Ctrl+C to stop)..."
    try {
        Invoke-Expression "docker compose $profiles logs -f"
    } finally {
        Write-Host "  Stopping containers..."
        Invoke-Expression "docker compose $profiles down"
        Write-Host "  Done."
    }
}

function Launch-Local {
    Write-Host ""

    # Check uv
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "  Installing uv..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    Write-Ok "uv installed"

    # Check pnpm
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        if (Get-Command npm -ErrorAction SilentlyContinue) {
            npm install -g pnpm
        } else {
            Exit-Fatal "pnpm/npm not found. Install Node.js: https://nodejs.org"
        }
    }
    Write-Ok "pnpm installed"

    Write-Host "  Installing dependencies..."
    uv sync --extra api --extra dev
    Push-Location m_flow-frontend; pnpm install --frozen-lockfile; Pop-Location
    Write-Ok "Dependencies installed"

    Write-Host "  Starting backend on :8000..."
    $backend = Start-Process -FilePath "uv" -ArgumentList "run python -m uvicorn m_flow.api.client:app --host 0.0.0.0 --port 8000" -PassThru -NoNewWindow

    Write-Host "  Starting frontend on :3000..."
    $frontend = Start-Process -FilePath "pnpm" -ArgumentList "dev" -WorkingDirectory "m_flow-frontend" -PassThru -NoNewWindow

    try {
        Write-Host "  Waiting for backend" -NoNewline
        if (Wait-ForHealth "http://localhost:8000/health" 90) { Write-Ok "Backend ready" }

        Write-Host "  Waiting for frontend" -NoNewline
        if (Wait-ForHealth "http://localhost:3000" 60) { Write-Ok "Frontend ready" }

        Show-ReadyBanner
        Start-Process "http://localhost:3000"

        Write-Host "  Press Ctrl+C to stop..."
        Wait-Process -Id $backend.Id
    } finally {
        if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
        if (-not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force }
        Write-Host "  Stopped."
    }
}

# ── Execute ──────────────────────────────────────────────────────────────

switch ($choice) {
    "1" { Launch-Docker "--profile ui" }
    "2" { Launch-Docker "--profile ui --profile neo4j" }
    "3" { Launch-Docker "--profile ui --profile postgres" }
    "4" { Launch-Local }
    "5" {
        Write-Host "  Available profiles: ui, mcp, neo4j, postgres, chromadb, redis"
        $custom = Read-Host "  Enter profiles (space-separated)"
        $profiles = ($custom.Split(" ") | ForEach-Object { "--profile $_" }) -join " "
        Launch-Docker $profiles
    }
    default { Exit-Fatal "Invalid choice: $choice" }
}
