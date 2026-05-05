# Define paths
$VenvRoot = "$HOME\.python_venvs"
$VenvPath = "$VenvRoot\arducam_capture"

Write-Host "--- Arducam Capture Environment Setup ---" -ForegroundColor Cyan

# 1. Ensure central venv folder exists
if (-not (Test-Path $VenvRoot)) {
    Write-Host "Creating central venv folder at $VenvRoot..." -ForegroundColor Gray
    New-Item -ItemType Directory -Path $VenvRoot -Force | Out-Null
}

# 2. Create venv if missing
if (-not (Test-Path $VenvPath)) {
    Write-Host "Virtual environment not found. Creating it now..." -ForegroundColor Yellow
    try {
        py -3.10 -m venv $VenvPath
        Write-Host "Successfully created venv at $VenvPath" -ForegroundColor Green
    } catch {
        Write-Error "Failed to create venv. Make sure Python 3.10 is installed."
        return
    }
} else {
    Write-Host "Existing virtual environment detected." -ForegroundColor Green
}

# 3. Activate
Write-Host "Activating..." -ForegroundColor Cyan
$ActivateScript = "$VenvPath\Scripts\Activate.ps1"

if (Test-Path $ActivateScript) {
    & $ActivateScript
    Write-Host "Environment 'arducam_capture' is now ACTIVE." -ForegroundColor DarkGreen
} else {
    Write-Error "Activation script missing! Try deleting the folder and running this again."
    return
}

# 4. Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Setup complete." -ForegroundColor Green
