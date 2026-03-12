param(
  [string]$BuildDir = "build",
  [string]$OutputDir = "results"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $OutputDir)) {
  New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$candidates = @(
  (Join-Path $BuildDir "transformer_kernels.exe"),
  (Join-Path $BuildDir "Release/transformer_kernels.exe"),
  (Join-Path $BuildDir "RelWithDebInfo/transformer_kernels.exe")
)

$binary = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $binary) {
  throw "Benchmark binary not found under $BuildDir. Configure and build the project first."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputPath = Join-Path $OutputDir "benchmark-$timestamp.txt"

& $binary | Tee-Object -FilePath $outputPath
Write-Host "Saved benchmark report to $outputPath"