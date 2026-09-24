param(
    [int]$PollSeconds = 30,
    [int]$FallbackAfterSeconds = 240
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "tmp"
$logPath = Join-Path $logDir "nightwatch.log"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
$teacherApi = "https://pulso-transmi.72-60-245-2.sslip.io"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Set-Location -LiteralPath $repo

$createdNew = $false
$mutex = New-Object System.Threading.Mutex($true, "Local\PulsoTransMiNightwatchCodex", [ref]$createdNew)
if (-not $createdNew) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) [INFO] Otra instancia ya esta activa; esta copia termina."
    exit 0
}

function Write-NightwatchLog {
    param([string]$Level, [string]$Message)
    $line = "$(Get-Date -Format o) [$Level] $Message"
    Add-Content -LiteralPath $logPath -Value $line
    Write-Output $line
}

function Import-DotEnv {
    $envPath = Join-Path $repo ".env"
    foreach ($line in Get-Content -LiteralPath $envPath) {
        if ($line -notmatch '^\s*[A-Za-z_][A-Za-z0-9_]*=') { continue }
        $key, $value = $line -split '=', 2
        $value = $value.Trim().Trim('"').Trim("'")
        Set-Item -Path ("Env:" + $key.Trim()) -Value $value
    }
}

function Invoke-Json {
    param([string]$Uri, [hashtable]$Headers = @{})
    $lastError = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            return Invoke-RestMethod -Uri $Uri -Headers $Headers -TimeoutSec 20
        }
        catch {
            $lastError = $_
            Start-Sleep -Seconds ([Math]::Min(5, $attempt * 2))
        }
    }
    throw $lastError
}

function Get-PublicHeaders {
    return @{ apikey = $env:SUPABASE_KEY }
}

function Get-AcceptedSubmissionCount {
    param([string]$CycleId)
    $base = $env:SUPABASE_URL.TrimEnd('/')
    $uri = "$base/rest/v1/submissions?select=submission_id&accepted=eq.true&cycle_id=eq.$CycleId"
    $rows = Invoke-Json -Uri $uri -Headers (Get-PublicHeaders)
    return ($rows | Measure-Object).Count
}

function Start-InferenceWorkflow {
    param([string]$CycleId)
    $env:GIT_CONFIG_COUNT = "1"
    $env:GIT_CONFIG_KEY_0 = "safe.directory"
    $env:GIT_CONFIG_VALUE_0 = "*"
    $output = & $gh workflow run inference.yml --repo kevin-nieto-callejas/pulso-transmi 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo disparar inference para $CycleId`: $output"
    }
    Write-NightwatchLog "ACTION" "Workflow de inference disparado para $CycleId."
}

function Start-LocalFallback {
    param([string]$CycleId)
    Write-NightwatchLog "ALERT" "Sin recibo tras $FallbackAfterSeconds s para $CycleId; ejecuto fallback local."
    $ingest = & $python src\ingest.py 2>&1
    Write-NightwatchLog "FALLBACK" ("ingest: " + (($ingest | Select-Object -Last 3) -join " | "))
    $infer = & $python src\infer.py 2>&1
    Write-NightwatchLog "FALLBACK" ("infer: " + (($infer | Select-Object -Last 8) -join " | "))
}

function Get-LatestRunSummary {
    $env:GIT_CONFIG_COUNT = "1"
    $env:GIT_CONFIG_KEY_0 = "safe.directory"
    $env:GIT_CONFIG_VALUE_0 = "*"
    $raw = & $gh run list --repo kevin-nieto-callejas/pulso-transmi --workflow inference --limit 1 `
        --json databaseId,status,conclusion,createdAt,headSha 2>$null
    if (-not $raw) { return "sin-run" }
    $run = $raw | ConvertFrom-Json | Select-Object -First 1
    $sha = if ($run.headSha) { $run.headSha.Substring(0, 7) } else { "sin-sha" }
    return "$($run.databaseId):$($run.status):$($run.conclusion):$sha"
}

function Write-OperationalSnapshot {
    param($Clock, $Cycle, [int]$Accepted)
    $headers = Get-PublicHeaders
    $base = $env:SUPABASE_URL.TrimEnd('/')
    $collector = Invoke-Json -Uri "$base/rest/v1/collector_runs?select=started_at,status,rows_ingested&order=started_at.desc&limit=1" -Headers $headers
    $champion = Invoke-Json -Uri "$base/rest/v1/model_versions?select=version_id,validation_metric&status=eq.champion&limit=1" -Headers $headers
    $drift = Invoke-Json -Uri "$base/rest/v1/drift_signals?select=detected_at,signal_type&order=detected_at.desc&limit=1" -Headers $headers

    $leaderboard = Invoke-Json -Uri "$teacherApi/v1/leaderboard?window=cumulative" `
        -Headers @{ Authorization = "Bearer $env:PULSO_API_KEY" }
    $me = $leaderboard.data | Where-Object { $_.display_name -like 'Kevin*' } | Select-Object -First 1
    $cycleId = if ($Cycle) { $Cycle.cycle_id } else { "none" }
    $collectorText = if ($collector) { "$($collector[0].status)/$($collector[0].rows_ingested)" } else { "none" }
    $championText = if ($champion) { "$($champion[0].version_id)/$([Math]::Round([double]$champion[0].validation_metric, 2))" } else { "none" }
    $driftText = if ($drift) { "$($drift[0].signal_type)@$($drift[0].detected_at)" } else { "none" }
    $rankText = if ($me) { "#$($me.rank)/$([Math]::Round([double]$me.accuracy, 2))/cov=$([Math]::Round(100 * [double]$me.coverage, 2))%" } else { "none" }
    $runText = Get-LatestRunSummary

    Write-NightwatchLog "HEARTBEAT" "tick=$($Clock.tick_number) virtual=$($Clock.virtual_now) cycle=$cycleId accepted=$Accepted collector=$collectorText champion=$championText rank=$rankText drift=$driftText run=$runText"
}

Import-DotEnv
if (-not (Test-Path -LiteralPath $python)) { throw "No existe el Python del proyecto: $python" }
if (-not (Test-Path -LiteralPath $gh)) { throw "No existe GitHub CLI: $gh" }

$cycles = @{}
$previousCode = $null
$previousTick = $null
$lastHeartbeat = [datetime]::MinValue
$lastEvaluation = [datetime]::MinValue
Write-NightwatchLog "START" "Nightwatch independiente iniciado (PID=$PID, poll=${PollSeconds}s, fallback=${FallbackAfterSeconds}s)."

try {
    while ($true) {
        try {
            $clock = Invoke-Json -Uri "$teacherApi/v1/clock"
            if ($previousCode -and $clock.code -ne $previousCode) {
                Write-NightwatchLog "ALERT" "Cambio de escenario: $previousCode -> $($clock.code)."
            }
            if ($null -ne $previousTick -and [int]$clock.tick_number -lt [int]$previousTick) {
                Write-NightwatchLog "ALERT" "El tick retrocedio: $previousTick -> $($clock.tick_number)."
            }
            $previousCode = $clock.code
            $previousTick = $clock.tick_number

            $cycle = $null
            try { $cycle = Invoke-Json -Uri "$teacherApi/v1/forecast-cycles/current" } catch { $cycle = $null }
            $accepted = 0
            if ($cycle) {
                $cycleId = [string]$cycle.cycle_id
                $accepted = Get-AcceptedSubmissionCount -CycleId $cycleId
                if (-not $cycles.ContainsKey($cycleId)) {
                    $cycles[$cycleId] = @{
                        firstSeen = Get-Date
                        workflowTriggered = $false
                        fallbackTriggered = $false
                        receiptLogged = $false
                    }
                    Write-NightwatchLog "CYCLE" "Ciclo abierto: $cycleId; cierra $($cycle.closes_at); targets=$($cycle.expected_predictions)."
                }
                $state = $cycles[$cycleId]
                if ($accepted -eq 0 -and -not $state.workflowTriggered) {
                    Start-InferenceWorkflow -CycleId $cycleId
                    $state.workflowTriggered = $true
                }
                if ($accepted -gt 0 -and -not $state.receiptLogged) {
                    Write-NightwatchLog "OK" "Submission aceptada para $cycleId (filas=$accepted)."
                    $state.receiptLogged = $true
                }
                $age = ((Get-Date) - [datetime]$state.firstSeen).TotalSeconds
                if ($accepted -eq 0 -and $age -ge $FallbackAfterSeconds -and -not $state.fallbackTriggered) {
                    $state.fallbackTriggered = $true
                    Start-LocalFallback -CycleId $cycleId
                    $accepted = Get-AcceptedSubmissionCount -CycleId $cycleId
                    if ($accepted -gt 0) {
                        Write-NightwatchLog "RECOVERED" "Fallback dejo submission aceptada para $cycleId."
                        $state.receiptLogged = $true
                    }
                }
            }

            if (((Get-Date) - $lastHeartbeat).TotalSeconds -ge 120) {
                Write-OperationalSnapshot -Clock $clock -Cycle $cycle -Accepted $accepted
                $lastHeartbeat = Get-Date
            }

            if (((Get-Date) - $lastEvaluation).TotalMinutes -ge 30) {
                # NumPy emite un DeprecationWarning conocido por stderr. En
                # PowerShell, con ErrorActionPreference=Stop, ese warning se
                # convierte en una excepcion aunque evaluate.py termine bien.
                # Se silencia solo esa categoria; cualquier fallo real conserva
                # exit code distinto de cero y queda registrado abajo.
                $evaluation = & $python -W "ignore::DeprecationWarning" src\evaluate.py 2>&1
                if ($LASTEXITCODE -ne 0) {
                    throw "evaluate.py termino con codigo $LASTEXITCODE`: $($evaluation -join ' | ')"
                }
                $decision = $evaluation | Select-String -Pattern 'MANTENER:|INVESTIGAR|REENTRENAR:' | Select-Object -Last 1
                $decisionText = if ($decision) { $decision.Line.Trim() } else { "evaluate.py termino sin decision legible" }
                Write-NightwatchLog "EVALUATE" $decisionText
                $lastEvaluation = Get-Date
            }
        }
        catch {
            Write-NightwatchLog "ERROR" $_.Exception.Message
        }
        Start-Sleep -Seconds $PollSeconds
    }
}
finally {
    Write-NightwatchLog "STOP" "Nightwatch detenido."
    try { $mutex.ReleaseMutex() } catch {}
    $mutex.Dispose()
}
