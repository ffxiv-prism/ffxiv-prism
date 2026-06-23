param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
    $PythonArgs = @('scripts\run_data_pipeline.py')
}

function Invoke-ProjectPython {
    param(
        [string]$PythonExe,
        [string[]]$ArgsToPass
    )

    Push-Location $ProjectRoot
    try {
        & $PythonExe @ArgsToPass
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

if ($env:FFXIV_PRISM_PYTHON) {
    if (-not (Test-Path -LiteralPath $env:FFXIV_PRISM_PYTHON -PathType Leaf)) {
        throw "FFXIV_PRISM_PYTHON points to a missing file: $env:FFXIV_PRISM_PYTHON"
    }
    Invoke-ProjectPython -PythonExe $env:FFXIV_PRISM_PYTHON -ArgsToPass $PythonArgs
}

$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) {
    Push-Location $ProjectRoot
    try {
        & $conda.Source run -n dataprocess python @PythonArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

$candidates = @(
    'D:\Anaconda\envs\dataprocess\python.exe',
    'C:\Softwares\Anaconda3\envs\dataprocess\python.exe',
    "$env:USERPROFILE\anaconda3\envs\dataprocess\python.exe",
    "$env:USERPROFILE\miniconda3\envs\dataprocess\python.exe"
)

foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        Invoke-ProjectPython -PythonExe $candidate -ArgsToPass $PythonArgs
    }
}

throw "Could not find the Anaconda environment named dataprocess. Set FFXIV_PRISM_PYTHON to the environment python.exe path, or make conda available in PATH."
