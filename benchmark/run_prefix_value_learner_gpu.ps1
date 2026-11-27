param(
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$PythonExecutable = 'C:\Users\ASUS\anaconda3\envs\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$image = 'sglang-prefix-research:cu130-v056post2-isolated'
$snapshot = '/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (Test-Path -LiteralPath $OutputRoot) { throw 'Choose a new evidence directory.' }
$root = (New-Item -ItemType Directory -Path $OutputRoot).FullName
$tag = 'study-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$conditions = @(
    @{ Policy='reuse_value'; Rate=0.03; Label='default' },
    @{ Policy='reuse_value_frozen'; Rate=0.03; Label='frozen' },
    @{ Policy='reuse_value'; Rate=0.10; Label='fast' }
)
$imageID = & $docker image inspect $image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Missing research image.' }
$order = @(0,1,2,2,1,0)
$paths = @()
$commit = & git -C $repo rev-parse HEAD
@{ Seed=1735; Concurrency=1; Capacity=2048; Image=$imageID; Source=$commit;
   Order=@($order | ForEach-Object { $conditions[$_] }); Model=$snapshot } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'plan.json') -Encoding utf8
for ($i=0; $i -lt $order.Count; $i++) {
    $case = $conditions[$order[$i]]
    $trial = Join-Path $root ("{0:D2}-{1}" -f ($i+1),$case.Label)
    $paths += $trial
    $trialTag = "$tag-$i"
    $name = "sglang-prefix-$($case.Policy)-2048-$trialTag"
    & (Join-Path $PSScriptRoot 'run_prefix_value_server.ps1') -Policy $case.Policy -CacheTokens 2048 -Port 30127 -Image $image -PythonExecutable /opt/sglang-research/bin/python -ModelPath $snapshot -LearningRate $case.Rate -TrialTag $trialTag
    try {
        & $PythonExecutable (Join-Path $PSScriptRoot 'run_prefix_value_trial.py') --policy $case.Policy --capacity 2048 --seed 1735 --url 'http://127.0.0.1:30127' --expected-learning-rate $case.Rate --output $trial
        if ($LASTEXITCODE -ne 0) { throw "Failed measurement at $trial" }
    } finally {
        try { & $docker logs $name *> "$trial-container.log" }
        finally { & $docker stop $name }
    }
}
& $PythonExecutable (Join-Path $PSScriptRoot 'summarize_prefix_value.py') @paths --output (Join-Path $root 'summary.json')
if ($LASTEXITCODE -ne 0) { throw 'Measurement verification failed.' }
