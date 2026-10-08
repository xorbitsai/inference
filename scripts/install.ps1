# Install and run Xinference in a persistent, isolated uv tool environment.
# irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex
$ErrorActionPreference = 'Stop'
function Get-Setting([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($value)) { return $Default }
    return $value
}
$pythonVersion = Get-Setting 'XINFERENCE_PYTHON' '3.12'
$backend = Get-Setting 'XINFERENCE_BACKEND' 'auto'
$start = Get-Setting 'XINFERENCE_START' '1'
$service = Get-Setting 'XINFERENCE_SERVICE' 'none'
$bindHost = Get-Setting 'XINFERENCE_HOST' '127.0.0.1'
$port = Get-Setting 'XINFERENCE_PORT' '9997'
if ($start -notin @('0', '1')) { throw 'XINFERENCE_START must be 0 or 1.' }
if ($service -notin @('none', 'system')) { throw 'Windows XINFERENCE_SERVICE must be none or system.' }
$portNumber = 0
if (-not [int]::TryParse($port, [ref]$portNumber) -or $portNumber -lt 1 -or $portNumber -gt 65535) {
    throw 'XINFERENCE_PORT must be between 1 and 65535.'
}
if ($service -eq 'system') {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this command in an Administrator PowerShell terminal to install a Windows service.'
    }
}
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
if ($env:XINFERENCE_HOME_DIR) { $env:UV_TOOL_DIR = $env:XINFERENCE_HOME_DIR }
$spec = Get-Setting 'XINFERENCE_PACKAGE' 'xinference'
if (-not $env:XINFERENCE_PACKAGE) {
    if ($env:XINFERENCE_EXTRAS) { $spec += "[$env:XINFERENCE_EXTRAS]" }
    if ($env:XINFERENCE_VERSION) {
        $version = $env:XINFERENCE_VERSION -replace '^v', ''
        if (-not $version) { throw 'XINFERENCE_VERSION must contain a version.' }
        $spec += "==$version"
    }
}
Write-Host "==> Installing $spec..."
& uv tool install --python $pythonVersion --torch-backend $backend $spec
if ($LASTEXITCODE -ne 0) { throw 'Xinference installation failed.' }
$toolDir = & uv tool dir
if ($LASTEXITCODE -ne 0) { throw 'Could not locate the uv tool environment.' }
$cli = Join-Path $toolDir 'xinference\Scripts\xinference.exe'
$server = Join-Path $toolDir 'xinference\Scripts\xinference-local.exe'
if (-not (Test-Path $server)) { throw "Xinference was not installed at $toolDir\xinference." }
Write-Host '==> Installed Xinference. Model engines use Xinference model environments; extras can also be preinstalled with XINFERENCE_EXTRAS.'
if ($service -eq 'system') {
    $serviceArgs = @('service', '--system', 'install', '--host', $bindHost, '--port', $port)
    if ($env:XINFERENCE_HOME) { $serviceArgs += @('--home', $env:XINFERENCE_HOME) }
    if ($start -eq '1') { $serviceArgs += '--start' }
    & $cli @serviceArgs
    if ($LASTEXITCODE -ne 0) { throw 'Xinference service installation or startup failed.' }
} elseif ($start -eq '1') {
    Write-Host "==> Starting Xinference on ${bindHost}:$port (Ctrl+C to stop)..."
    & $server --host $bindHost --port $port
    if ($LASTEXITCODE -ne 0) { throw 'Xinference server exited with an error.' }
} else {
    Write-Host "==> Start the server: & '$server' --host $bindHost --port $port"
}
