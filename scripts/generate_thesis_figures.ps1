$pythonCandidates = @(
    'E:\soft\anaconda\envs\bishe\python.exe',
    'python'
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq 'python') {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $pythonExe = $cmd.Source
            break
        }
    } elseif (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    throw '未找到可用的 Python 解释器，无法生成论文插图。'
}

$scriptPath = Join-Path $PSScriptRoot 'generate_thesis_figures.py'
& $pythonExe $scriptPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}