# Fix Git Large Files Issue
# Removes large files from Git tracking and adds them to .gitignore

Write-Host "Fixing Git large files issue..." -ForegroundColor Green

# Step 1: Remove large files from Git tracking
Write-Host "Step 1: Removing large files from Git tracking..." -ForegroundColor Yellow

# Remove the large dataset files from Git tracking
git rm --cached -r datasets/hl_cead/ 2>$null
git rm --cached -r data/ 2>$null
git rm --cached -r models/ 2>$null
git rm --cached -r results/ 2>$null
git rm --cached -r evaluation_results/ 2>$null
git rm --cached -r engine_sounds/ 2>$null
git rm --cached -r logs/ 2>$null
git rm --cached -r notebooks/ 2>$null
git rm --cached -r src/ 2>$null
git rm --cached -r tests/ 2>$null

# Remove specific large files
git rm --cached "best_ultra_compact_model.pth" 2>$null
git rm --cached "ultra_compact_detailed_analysis.png" 2>$null
git rm --cached "ultra_compact_evaluation_metrics.png" 2>$null
git rm --cached "ultra_compact_training_curves.png" 2>$null
git rm --cached "comprehensive_engine_comparison.png" 2>$null
git rm --cached "simple_engine_test_visualization.png" 2>$null
git rm --cached "stftsc_algorithm_flowchart.png" 2>$null
git rm --cached "stftsc_energy_beam_visualization.png" 2>$null

Write-Host "Large files removed from Git tracking" -ForegroundColor Green

# Step 2: Create or update .gitignore
Write-Host "Step 2: Updating .gitignore..." -ForegroundColor Yellow

$gitignoreContent = @"
# Large files and directories
datasets/
data/
models/
results/
evaluation_results/
engine_sounds/
logs/
notebooks/
src/
tests/

# Model files
*.pth
*.pkl
*.h5
*.hdf5

# Large image files
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.tiff

# Audio files
*.wav
*.mp3
*.flac
*.aac
*.ogg

# Video files
*.mp4
*.avi
*.mov
*.mkv

# Data files
*.zip
*.tar
*.gz
*.rar
*.7z

# Log files
*.log
*.out

# Temporary files
*.tmp
*.temp
*~

# OS files
.DS_Store
Thumbs.db

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Docker
.dockerignore
"@

# Check if .gitignore exists, create it if not
if (-not (Test-Path .gitignore)) {
    Write-Host "Creating .gitignore file..." -ForegroundColor Yellow
    $gitignoreContent | Out-File -FilePath .gitignore -Encoding UTF8
    Write-Host "Created .gitignore file." -ForegroundColor Green
} else {
    Write-Host ".gitignore already exists. Updating..." -ForegroundColor Yellow
    $currentGitignoreContent = Get-Content .gitignore -Raw
    if ($currentGitignoreContent -notmatch $gitignoreContent) {
        Write-Host "Updating .gitignore..." -ForegroundColor Yellow
        $gitignoreContent | Out-File -FilePath .gitignore -Encoding UTF8 -Append
        Write-Host "Updated .gitignore." -ForegroundColor Green
    } else {
        Write-Host ".gitignore is already up-to-date." -ForegroundColor Green
    }
}

Write-Host "Git large files issue fix complete." -ForegroundColor Green 