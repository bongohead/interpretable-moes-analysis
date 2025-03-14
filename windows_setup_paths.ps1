# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Get the project root directory (interpretable-moes)
$projectPath = $scriptDir

# Path to the .venv folder (sibling to the script)
$venvPath = Join-Path $scriptDir ".venv"

# Verify the .venv directory exists
if (-Not (Test-Path $venvPath)) {
    Write-Error "Virtual environment directory not found at: $venvPath"
    exit 1
}

# Activate the virtual environment and get its site-packages directory
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
if (-Not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found at: $pythonExe"
    exit 1
}

$sitePkgDir = & $pythonExe -c "import site; print(site.getsitepackages()[0])"

# Create the .pth file in the virtual environment's site-packages directory
$projectPath | Out-File -FilePath "$sitePkgDir\add_path_analysis.pth" -Encoding utf8

Write-Host "Added path '$projectPath' to site-packages in virtual environment at '$venvPath'"