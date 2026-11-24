# release.ps1 - publish a GitHub release of Quran Recitation Tracker. PowerShell 5.1, ASCII only.
# Run from anywhere inside the repo, on branch main with everything committed:
#   .\scripts\release.ps1 -Tag v0.2.0              build the cpu + gpu zips, push main, tag, publish the release
#   .\scripts\release.ps1 -Tag v0.2.0 -SkipBuild   reuse the zips already in backend\dist
#   .\scripts\release.ps1 -Tag v0.2.0 -Draft       create a draft (check it on GitHub, then press Publish)
#   .\scripts\release.ps1 -Tag v0.2.0 -NoPackages  source-only release
# Needs: GitHub CLI signed in (gh auth login), kit\offline_cache filled once (python backend\kit\fetch_offline.py),
# and the speech model in backend\whisper-quran-ct2-muq or backend\model.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [switch]$SkipBuild,
    [switch]$Draft,
    [switch]$NoPackages
)
$ErrorActionPreference = 'Stop'
function Die([string]$Msg) { Write-Host ('  ' + $Msg) -ForegroundColor Red; exit 1 }
function Step([string]$Msg) { Write-Host ('* ' + $Msg) -ForegroundColor Cyan }

$Root = (git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
Set-Location -LiteralPath $Root

# ---- checks (nothing is changed before all of them pass)
if ($Tag -notmatch '^v\d+\.\d+\.\d+$') { Die 'The tag must look like v1.2.3' }
$gh = (Get-Command gh -ErrorAction SilentlyContinue).Source
if (-not $gh -and (Test-Path 'C:\Program Files\GitHub CLI\gh.exe')) { $gh = 'C:\Program Files\GitHub CLI\gh.exe' }
if (-not $gh) { Die 'GitHub CLI not found. Install it: winget install GitHub.cli' }
& $gh auth status *> $null
if ($LASTEXITCODE -ne 0) { Die 'GitHub CLI is not signed in. Run: gh auth login' }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne 'main') { Die ('Releases are made from main (you are on ' + $branch + ').') }
if (git status --porcelain) { Die 'There are uncommitted changes. Commit or stash them first.' }
git fetch --quiet --tags origin
if ($LASTEXITCODE -ne 0) { Die 'Could not reach GitHub (git fetch failed).' }
if (git tag --list $Tag) { Die ('Tag ' + $Tag + ' already exists.') }
$behind = (git rev-list --count HEAD..origin/main).Trim()
if ($behind -ne '0') { Die ('main is ' + $behind + ' commit(s) behind GitHub. Run: git pull') }

$app = (Select-String -Path 'backend\server.py' -Pattern '^APP_VERSION = "([^"]+)"').Matches[0].Groups[1].Value
$zips = @()
if (-not $NoPackages) {
    $zips = @(('backend\dist\QuranTracker-' + $app + '-cpu.zip'), ('backend\dist\QuranTracker-' + $app + '-gpu.zip'))
    if (-not $SkipBuild) {
        $model = @('backend\whisper-quran-ct2-muq', 'backend\model') | Where-Object { Test-Path (Join-Path $_ 'model.bin') } | Select-Object -First 1
        if (-not $model) { Die 'Speech model not found (backend\whisper-quran-ct2-muq or backend\model).' }
        $py = $env:QT_PYTHON
        if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
        if (-not $py) { Die 'Python not found (set QT_PYTHON to python.exe).' }
        Step ('Building the packages for app ' + $app + ' ...')
        & $py 'backend\kit\make_package.py' --model $model --edition cpu gpu --out 'backend\dist'
        if ($LASTEXITCODE -ne 0) { Die 'Building the packages failed (see above).' }
    }
    foreach ($z in $zips) { if (-not (Test-Path $z)) { Die ('Missing package: ' + $z) } }
}

# ---- publish
Step 'Pushing main ...'
git push origin main
if ($LASTEXITCODE -ne 0) { Die 'git push failed.' }
Step ('Tagging ' + $Tag + ' ...')
git tag -a $Tag -m ('Quran Recitation Tracker ' + $Tag + ' (app ' + $app + ')')
git push origin $Tag
if ($LASTEXITCODE -ne 0) { Die 'Pushing the tag failed.' }

$notes = Join-Path $env:TEMP ('qt_release_' + $Tag + '.md')
$text = @(
    ('App version ' + $app + '.'),
    '',
    '### Download',
    ('- **QuranTracker-' + $app + '-gpu.zip**: PCs with an NVIDIA graphics card (faster).'),
    ('- **QuranTracker-' + $app + '-cpu.zip**: any other Windows 10/11 64-bit PC.'),
    '',
    'Both install without internet: extract the whole folder (e.g. to C:\QuranTracker) and double-click Start.bat.',
    ''
)
if ($NoPackages) { $text = @(('App version ' + $app + '. Source-only release.'), '') }
Set-Content -LiteralPath $notes -Value $text -Encoding UTF8
$args_ = @('release', 'create', $Tag) + $zips + @('--title', ('Quran Recitation Tracker ' + $Tag), '--notes-file', $notes, '--generate-notes', '--verify-tag')
if ($Draft) { $args_ += '--draft' }
Step 'Creating the GitHub release (uploading the packages can take several minutes) ...'
& $gh @args_
if ($LASTEXITCODE -ne 0) { Die ('Creating the release failed. The tag is pushed; retry with: gh release create ' + $Tag + ' ...') }
Remove-Item -LiteralPath $notes -ErrorAction SilentlyContinue
Step 'Done.'
