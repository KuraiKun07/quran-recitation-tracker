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

# ---- GitHub refuses release files over 2 GiB: split bigger zips into parts + a Join-*.bat that rebuilds them
$PartBytes = 1900000000
$assets = @()
$joinNotes = @()
$relDir = Join-Path $Root ('backend\dist\release_' + $Tag)
if ($zips.Count -gt 0) {
    if (Test-Path -LiteralPath $relDir) { Remove-Item -LiteralPath $relDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $relDir | Out-Null
}
foreach ($z in $zips) {
    $full = (Resolve-Path -LiteralPath $z).Path
    $name = Split-Path -Leaf $full
    $size = (Get-Item -LiteralPath $full).Length
    if ($size -le $PartBytes) { $assets += $full; continue }
    Step ('Splitting ' + $name + ' (' + [math]::Round($size / 1GB, 2) + ' GB) for GitHub ...')
    $hash = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
    $in = [System.IO.File]::OpenRead($full)
    $buf = New-Object byte[] (16MB)
    $parts = @()
    try {
        $i = 0
        while ($in.Position -lt $in.Length) {
            $i++
            $pname = $name + '.' + $i.ToString('000')
            $out = [System.IO.File]::Create((Join-Path $relDir $pname))
            try {
                $left = $PartBytes
                while ($left -gt 0) {
                    $n = $in.Read($buf, 0, [int][math]::Min($buf.Length, $left))
                    if ($n -le 0) { break }
                    $out.Write($buf, 0, $n)
                    $left -= $n
                }
            } finally { $out.Close() }
            $parts += $pname
        }
    } finally { $in.Close() }
    $edition = ($name -replace '^QuranTracker-.*-', '') -replace '\.zip$', ''
    $bat = 'Join-' + $edition + '-zip.bat'
    $lines = @('@echo off', 'cd /d "%~dp0"', ('echo Joining the ' + $parts.Count + ' parts into ' + $name + ' ...'),
        ('copy /b "' + ($parts -join '" + "') + '" "' + $name + '" >nul'),
        'if errorlevel 1 (echo Failed: put all the parts in the same folder as this file. & pause & exit /b 1)',
        ('echo Done: ' + $name + ' is ready. Extract it like any zip.'), 'pause')
    [System.IO.File]::WriteAllText((Join-Path $relDir $bat), (($lines -join "`r`n") + "`r`n"), [System.Text.Encoding]::ASCII)
    foreach ($p in $parts) { $assets += (Join-Path $relDir $p) }
    $assets += (Join-Path $relDir $bat)
    $joinNotes += ('- **' + $name + '** is split in ' + $parts.Count + ' parts (GitHub limit). Download ' + (($parts | ForEach-Object { '`' + $_ + '`' }) -join ', ') +
        ' and `' + $bat + '` into one folder and double-click the .bat: it rebuilds the zip (SHA-256 ' + $hash + ').')
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
) + $joinNotes + @('')
if ($NoPackages) { $text = @(('App version ' + $app + '. Source-only release.'), '') }
Set-Content -LiteralPath $notes -Value $text -Encoding UTF8
$args_ = @('release', 'create', $Tag) + $assets + @('--title', ('Quran Recitation Tracker ' + $Tag), '--notes-file', $notes, '--generate-notes', '--verify-tag')
if ($Draft) { $args_ += '--draft' }
Step 'Creating the GitHub release (uploading the packages can take several minutes) ...'
& $gh @args_
if ($LASTEXITCODE -ne 0) { Die ('Creating the release failed. The tag is pushed; retry with: gh release create ' + $Tag + ' ...') }
Remove-Item -LiteralPath $notes -ErrorAction SilentlyContinue
Step 'Done.'
