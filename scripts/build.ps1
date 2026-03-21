param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Release",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [string]$Label,
        [string[]]$CommandParts
    )

    Write-Host ">> $Label"
    & $CommandParts[0] @($CommandParts[1..($CommandParts.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build"
$configureArgs = @("cmake", "-S", $root, "-B", $buildDir)

if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    $resolvedPython = if (Test-Path $PythonExe) { (Resolve-Path $PythonExe).Path } else { $PythonExe }
    Write-Host "Using Python: $resolvedPython"

    & $resolvedPython -c "import torch; print(torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch is not installed in '$resolvedPython'. Run: `"$resolvedPython`" -m pip install torch"
    }

    $configureArgs += "-DPython_EXECUTABLE=$resolvedPython"
}
else {
    Write-Warning "Python interpreter not specified. CMake will select one automatically."
}

Invoke-CheckedCommand -Label "Configure CMake" -CommandParts $configureArgs
Invoke-CheckedCommand -Label "Build" -CommandParts @("cmake", "--build", $buildDir, "--config", $Config, "--parallel")
Invoke-CheckedCommand -Label "Install" -CommandParts @("cmake", "--install", $buildDir, "--config", $Config, "--prefix", $root)

Write-Host "Build complete."
Write-Host "Torch library installed under: $root\python\rt2d"
