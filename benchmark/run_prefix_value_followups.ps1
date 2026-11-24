param(
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$PythonExecutable = 'C:\Users\ASUS\anaconda3\envs\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$dockerPath = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$sourcePath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new output root; existing evidence is never overwritten.' }
$dirty = & git -C $sourcePath status --porcelain
if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Commit the source before starting the matrix.' }
$sourceCommit = & git -C $sourcePath rev-parse HEAD
$expectedImage = 'sha256:491b47f3679caef58cdc9148d4ab4a5a08e31468c1aeb53630e6952350f29d9a'
$plan = @()
foreach ($policy in @('reuse_value_frozen', 'indexed_lru', 'lru', 'reuse_value')) {
    $plan += [pscustomobject]@{ Name="sequential-r02-$policy"; Policy=$policy; Concurrency=1 }
}
foreach ($policy in @('lru', 'reuse_value', 'reuse_value_frozen', 'indexed_lru')) {
    $plan += [pscustomobject]@{ Name="wave4-r01-$policy"; Policy=$policy; Concurrency=4 }
}
$runRoot = (New-Item -ItemType Directory -Path $OutputRoot).FullName
@{ SourceCommit=$sourceCommit; Image=$expectedImage; Capacity=2048; Seed=1729; Trials=$plan } |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'plan.json') -Encoding utf8
$statusPath = Join-Path $runRoot 'status.json'
foreach ($entry in $plan) {
    $containerName = "sglang-prefix-$($entry.Policy)-2048"
    $trialPath = Join-Path $runRoot $entry.Name
    $imageId = & $dockerPath inspect --format '{{.Image}}' $containerName
    if ($LASTEXITCODE -ne 0 -or $imageId -ne $expectedImage) { throw "Unexpected image for $containerName" }
    @{ State='running'; Trial=$entry.Name; SourceCommit=$sourceCommit; StartedAt=(Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
    try {
        & $dockerPath start $containerName
        if ($LASTEXITCODE -ne 0) { throw "Failed to start $containerName" }
        & $PythonExecutable (Join-Path $PSScriptRoot 'run_prefix_value_trial.py') `
            --policy $entry.Policy --capacity 2048 --seed 1729 --concurrency $entry.Concurrency --output $trialPath
        if ($LASTEXITCODE -ne 0) { throw "Trial failed: $($entry.Name)" }
    } catch {
        @{ State='failed'; Trial=$entry.Name; Error=$_.Exception.Message } |
            ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
        throw
    } finally {
        & $dockerPath logs $containerName 2>&1 | Set-Content -LiteralPath (Join-Path $runRoot "$($entry.Name)-container.log") -Encoding utf8
        & $dockerPath stop $containerName
        if ($LASTEXITCODE -ne 0) { throw "Failed to stop $containerName; inspect GPU before continuing" }
    }
}
$currentCommit = & git -C $sourcePath rev-parse HEAD
if ($currentCommit -ne $sourceCommit) { throw 'Source commit changed during the experiment.' }
$trials = @($plan | ForEach-Object { Join-Path $runRoot $_.Name })
& $PythonExecutable (Join-Path $PSScriptRoot 'summarize_prefix_value.py') @trials --output (Join-Path $runRoot 'summary.json')
if ($LASTEXITCODE -ne 0) { throw 'Completed trial validation failed' }
@{ State='complete'; Trials=$plan.Count; SourceCommit=$sourceCommit; FinishedAt=(Get-Date).ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
