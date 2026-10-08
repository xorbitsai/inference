# Install and run Xinference in a persistent, isolated uv tool environment.
# irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex
$ErrorActionPreference = 'Stop'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$compatible = $false
if ($uvCommand) {
    $toolHelp = & uv tool install --help
    $compatible = $LASTEXITCODE -eq 0 -and [bool]($toolHelp -match '--torch-backend')
}
if (-not $compatible) {
    Write-Host '==> Installing a compatible uv...'
    & ([scriptblock]::Create((Invoke-RestMethod 'https://astral.sh/uv/install.ps1')))
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv was not found after installation.' }
$toolHelp = & uv tool install --help
if ($LASTEXITCODE -ne 0 -or -not ($toolHelp -match '--torch-backend')) {
    throw 'Upgrade uv to a version supporting uv tool install --torch-backend (tested with 0.11.26).'
}
# The standalone driver works even when the installed release predates it.
$installer = if ($PSScriptRoot) { Join-Path $PSScriptRoot 'manage_install.py' } else { '' }
$temporary = $false
if (-not $installer -or -not (Test-Path $installer)) {
    $installer = [IO.Path]::GetTempFileName()
    $temporary = $true
    $url = if ($env:XINFERENCE_INSTALLER_URL) { $env:XINFERENCE_INSTALLER_URL } else { 'https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/manage_install.py' }
    Invoke-WebRequest $url -OutFile $installer
}
try {
    & uv run --no-project --no-config --python 3.12 python $installer
    if ($LASTEXITCODE -ne 0) { throw 'Xinference installation or upgrade failed.' }
} finally {
    if ($temporary) { Remove-Item $installer -ErrorAction SilentlyContinue }
}
