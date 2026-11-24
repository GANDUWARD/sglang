param(
    [ValidateSet('lru', 'lfu', 'indexed_lru', 'reuse_value', 'reuse_value_frozen')]
    [string]$Policy = 'lru',
    [ValidateRange(1024, 16384)][int]$CacheTokens = 4096,
    [int]$Port = 30000,
    [string]$Image = 'lmsysorg/sglang:v0.5.6.post2',
    [string]$PythonExecutable = 'python3',
    [string]$ModelPath = 'Qwen/Qwen3-0.6B',
    [switch]$DeterministicInference
)
$ErrorActionPreference = 'Stop'
$dockerPath = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$sourcePath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$containerName = "sglang-prefix-$Policy-$CacheTokens"
$chunkedPrefillSize = 512
if ($DeterministicInference) {
    $containerName += '-deterministic'
    # v0.5.6.post2 requires 4096-token prefill truncation alignment in this mode.
    # A smaller budget can leave long prompts permanently unadmitted.
    $chunkedPrefillSize = 4096
}
$arguments = @(
    'run', '-d', '--name', $containerName, '--gpus', 'all', '--shm-size', '2g',
    '-p', "127.0.0.1:${Port}:30000",
    '--mount', "type=bind,source=$sourcePath,target=/workspace,readonly",
    '-v', 'vllm-hf-cache:/root/.cache/huggingface:ro',
    '-e', 'PYTHONPATH=/workspace/python', '-e', 'HF_HUB_OFFLINE=1',
    '--entrypoint', $PythonExecutable, $Image,
    '-m', 'sglang.launch_server', '--model-path', $ModelPath,
    '--host', '0.0.0.0', '--port', '30000', '--context-length', '2048',
    '--max-total-tokens', "$CacheTokens", '--max-running-requests', '4',
    '--chunked-prefill-size', "$chunkedPrefillSize", '--mem-fraction-static', '0.65',
    '--disable-cuda-graph', '--attention-backend', 'triton',
    '--radix-eviction-policy', $Policy, '--enable-metrics', '--random-seed', '1729'
)
if ($DeterministicInference) { $arguments += '--enable-deterministic-inference' }
& $dockerPath @arguments
if ($LASTEXITCODE -ne 0) { throw 'Server container creation failed' }
Write-Output "Started $containerName; verify readiness and logs before replay."
