# run.ps1 - one-click launcher for Quran Tracker: checks, first-run setup, start. PowerShell 5.1, ASCII only.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }
try { $Host.UI.RawUI.WindowTitle = 'Quran Tracker' } catch { }

$Setup = $PSScriptRoot
$Root = Split-Path -Parent $Setup
$App = Join-Path $Root 'app'
$Back = Join-Path $App 'backend'
$Web = Join-Path $App 'client-web'
$Model = Join-Path $Back 'model'
$RtDir = Join-Path $Root 'runtime'
$PyDir = Join-Path $RtDir 'python'
$Py = Join-Path $PyDir 'python.exe'
$Logs = Join-Path $Root 'logs'
$LockFile = Join-Path $Setup 'requirements.lock.txt'
$CudaFile = Join-Path $Setup 'requirements-cuda.txt'
$Marker = Join-Path $RtDir 'ready.txt'
# Offline editions carry everything in setup\offline (Python, VC++ runtime, wheels); no internet is used then.
$Offline = Join-Path $Setup 'offline'
$Wheels = Join-Path $Offline 'wheels'
$IsOffline = Test-Path -LiteralPath $Wheels
$HasCudaWheels = $IsOffline -and (@(Get-ChildItem -LiteralPath $Wheels -Filter 'nvidia_cudnn*.whl').Count -gt 0)
$PipSource = @()
if ($IsOffline) { $PipSource = @('--no-index', '--find-links', $Wheels) }
$BasePort = 8000
if ($env:QT_PORT) { $BasePort = [int]$env:QT_PORT }

. (Join-Path $Setup 'lib.ps1')

function Fail {
    param([string]$Msg, [string]$Hint = '')
    Write-Host ''
    Write-Host ('  ' + $Msg) -ForegroundColor Red
    if ($Hint) { Write-Host ('  ' + $Hint) -ForegroundColor Yellow }
    Write-Host '  More details are in the "logs" folder next to Start.bat.' -ForegroundColor Yellow
    Write-Log ('FAIL: ' + $Msg + ' | ' + $Hint) 'ERROR'
    exit 1
}

try {
    Write-Host ''
    Write-Host '  Quran Tracker' -ForegroundColor Green
    Write-Host '  -------------' -ForegroundColor Green

    # ---- 0. logs folder (also proves the folder is writable)
    if (-not (Test-Path -LiteralPath $Logs)) { New-Item -ItemType Directory -Path $Logs -Force | Out-Null }
    $script:LogFile = Join-Path $Logs 'setup.log'
    Write-Log ('start; root=' + $Root + '; PS ' + $PSVersionTable.PSVersion)

    # ---- 1. basic checks
    if (-not [Environment]::Is64BitOperatingSystem -or [Environment]::OSVersion.Version.Major -lt 10) {
        Fail 'This app needs 64-bit Windows 10 or Windows 11.'
    }
    if (-not (Test-Writable -Dir $Root)) {
        Fail 'This folder is read-only.' 'Move the whole folder to a normal place such as C:\QuranTracker and try again.'
    }
    if (Test-PathTooLong -Path $Root) {
        Fail 'This folder path is too long.' 'Move the whole folder to a short place such as C:\QuranTracker and try again.'
    }
    foreach ($f in @((Join-Path $Back 'server.py'), (Join-Path $Back 'quran.db'), (Join-Path $Back 'cudaprep.py'),
                     (Join-Path $Model 'model.bin'), (Join-Path $Model 'tokenizer.json'),
                     (Join-Path $Web 'index.html'), $LockFile, $CudaFile)) {
        if (-not (Test-Path -LiteralPath $f)) {
            Fail 'Some app files are missing.' 'Please unzip the complete folder again (Extract All) and start from there.'
        }
    }

    # ---- 2. already running?
    $Port = $BasePort
    if (Test-PortOpen -Port $Port) {
        if (Test-HttpOk -Url ('http://127.0.0.1:' + $Port + '/')) {
            Say 'Quran Tracker is already running. Opening it in your browser.' 'Green'
            Start-Process ('http://localhost:' + $Port + '/')
            exit 0
        }
        $found = $false
        for ($p = $BasePort + 1; $p -le $BasePort + 10; $p++) {
            if (-not (Test-PortOpen -Port $p)) { $Port = $p; $found = $true; break }
        }
        if (-not $found) { Fail 'Another program is using the network port this app needs.' 'Restart the computer and try again.' }
        Say ('Port ' + $BasePort + ' is busy, using port ' + $Port + ' instead.') 'Yellow'
    }

    # ---- 3. what hardware do we have?
    $gpu = Get-NvidiaGpu
    $useGpu = Test-GpuSuitable -Gpu $gpu
    if ($useGpu -and $IsOffline -and -not $HasCudaWheels) {
        Say ('NVIDIA graphics card found (' + $gpu.Name + '), but this is the processor edition of the app.') 'Yellow'
        Say 'It works well at a calm reciting pace. The GPU edition gives faster results on this PC.' 'Yellow'
        $useGpu = $false
    } elseif ($useGpu) {
        Say ('NVIDIA graphics card found: ' + $gpu.Name + '. It will be used for faster results.') 'Green'
    } elseif ($null -ne $gpu) {
        Say ('NVIDIA card found (' + $gpu.Name + ') but its driver (' + $gpu.Driver + ') is too old or the card too small.') 'Yellow'
        Say 'Using the processor instead. Updating the NVIDIA driver can make results faster.' 'Yellow'
    } else {
        Say 'No NVIDIA graphics card found. Using the processor (works well at a calm reciting pace).' 'White'
    }

    # ---- 4. first-run setup (only what is missing)
    $lockHash = (Get-FileHash -LiteralPath $LockFile -Algorithm SHA256).Hash
    $cudaHash = (Get-FileHash -LiteralPath $CudaFile -Algorithm SHA256).Hash
    $state = Read-Marker -Path $Marker
    $needBase = (-not (Test-Path -LiteralPath $Py)) -or ($state['lock'] -ne $lockHash)
    $needCuda = $useGpu -and ($state['cuda'] -ne $cudaHash)
    $missingVc = @(Get-MissingVcRuntime)

    if ($needBase -or $needCuda -or $missingVc.Count -gt 0) {
        Write-Host ''
        if ($IsOffline) {
            Say 'First-time setup from the included files (no internet needed). It takes a few minutes. Please keep this window open.' 'Cyan'
        } else {
            Say 'First-time setup. This needs the internet once and takes a few minutes. Please keep this window open.' 'Cyan'
        }
        $needGB = 1.0
        if ($needCuda) { $needGB = 3.5 }
        $free = Get-FreeGB -Path $Root
        if (($null -ne $free) -and ($free -lt $needGB)) {
            Fail ('Not enough free disk space (' + $free + ' GB free, about ' + $needGB + ' GB needed).') 'Free some space or move the folder to another drive.'
        }
        if (-not (Test-Path -LiteralPath $RtDir)) { New-Item -ItemType Directory -Path $RtDir -Force | Out-Null }
        $tmp = Join-Path $RtDir 'tmp'
        if (-not (Test-Path -LiteralPath $tmp)) { New-Item -ItemType Directory -Path $tmp -Force | Out-Null }
        $pipLog = Join-Path $Logs 'pip.log'

        # 4a. Microsoft Visual C++ runtime (needed by the speech engine; most PCs already have it)
        if ($missingVc.Count -gt 0) {
            Say ('Installing Microsoft Visual C++ runtime (missing: ' + ($missingVc -join ', ') + '). Windows may ask for permission.') 'Cyan'
            $exe = Join-Path $Offline 'vc_redist.x64.exe'
            if (-not (Test-Path -LiteralPath $exe)) {
                $exe = Join-Path $tmp 'vc_redist.x64.exe'
                if (-not (Get-File -Url 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -Dest $exe)) {
                    Fail 'Could not download the Microsoft Visual C++ runtime.' 'Check your internet connection and try again.'
                }
            }
            try {
                $pr = Start-Process -FilePath $exe -ArgumentList '/install', '/quiet', '/norestart' -Verb RunAs -Wait -PassThru
            } catch {
                Fail 'The Visual C++ runtime was not installed (permission refused).' 'Start again and click Yes when Windows asks.'
            }
            if (@(0, 1638, 3010) -notcontains $pr.ExitCode) {
                Fail ('The Visual C++ runtime installer failed (code ' + $pr.ExitCode + ').') 'Install it manually from https://aka.ms/vs/17/release/vc_redist.x64.exe and start again.'
            }
            if ($pr.ExitCode -eq 3010) { Say 'A restart is recommended after setup finishes.' 'Yellow' }
        }

        # 4b. Python (a private copy inside this folder; nothing is installed on the PC)
        if ($needBase -and (-not (Test-Path -LiteralPath $Py))) {
            $local = @(Get-ChildItem -LiteralPath $Offline -Filter 'python-*-embed-amd64.zip' -ErrorAction SilentlyContinue)
            if ($local.Count -gt 0) {
                Say 'Preparing Python ...' 'Cyan'
                $zip = $local[0].FullName
                Write-Log ('python from ' + $local[0].Name)
            } else {
                Say 'Downloading Python ...' 'Cyan'
                $zip = Join-Path $tmp 'python-embed.zip'
                $got = $false
                foreach ($v in @('3.12.10', '3.12.9', '3.12.8')) {
                    $u = 'https://www.python.org/ftp/python/' + $v + '/python-' + $v + '-embed-amd64.zip'
                    if (Get-File -Url $u -Dest $zip) { $got = $true; Write-Log ('python ' + $v); break }
                }
                if (-not $got) { Fail 'Could not download Python.' 'Check your internet connection (or antivirus/firewall) and try again.' }
            }
            if (Test-Path -LiteralPath $PyDir) { Remove-Item -LiteralPath $PyDir -Recurse -Force }
            Expand-Archive -LiteralPath $zip -DestinationPath $PyDir -Force
        }
        if (Test-Path -LiteralPath $PyDir) { Write-PthFile -PyDir $PyDir }

        # 4c. pip
        $chk = Invoke-Native -Exe $Py -ArgList @('-m', 'pip', '--version')
        $pipWhl = @(Get-ChildItem -LiteralPath $Wheels -Filter 'pip-*.whl' -ErrorAction SilentlyContinue)
        if ($chk.Code -ne 0 -and $pipWhl.Count -gt 0) {
            Say 'Preparing the installer tool (pip) ...' 'Cyan'
            # a wheel is a zip: unpacking it into site-packages installs pip (enough for python -m pip)
            $site = Join-Path $PyDir 'Lib\site-packages'
            if (-not (Test-Path -LiteralPath $site)) { New-Item -ItemType Directory -Path $site -Force | Out-Null }
            $pz = Join-Path $tmp 'pip.zip'
            Copy-Item -LiteralPath $pipWhl[0].FullName -Destination $pz -Force
            Expand-Archive -LiteralPath $pz -DestinationPath $site -Force
            $r = Invoke-Native -Exe $Py -ArgList @('-m', 'pip', '--version')
            Write-Log $r.Output
            if ($r.Code -ne 0) { Fail 'Could not set up the installer tool (pip).' 'Try again; if it repeats, send the logs folder to the person who shared the app.' }
        } elseif ($chk.Code -ne 0) {
            Say 'Preparing the installer tool (pip) ...' 'Cyan'
            $gp = Join-Path $tmp 'get-pip.py'
            if (-not (Get-File -Url 'https://bootstrap.pypa.io/get-pip.py' -Dest $gp)) {
                Fail 'Could not download the installer tool (pip).' 'Check your internet connection and try again.'
            }
            $r = Invoke-Native -Exe $Py -ArgList @($gp, '--no-warn-script-location')
            Write-Log $r.Output
            if ($r.Code -ne 0) { Fail 'Could not set up the installer tool (pip).' 'Try again; if it repeats, send the logs folder to the person who shared the app.' }
        }

        # 4d. app packages
        if ($needBase) {
            Say 'Installing app components (a few hundred MB) ...' 'Cyan'
            $rc = Invoke-Pip -Py $Py -PipArgs (@('install', '--only-binary=:all:') + $PipSource + @('-r', $LockFile)) -PipLog $pipLog
            if ($rc -ne 0) { Fail 'Installing the app components failed.' 'Check your internet connection (or antivirus/firewall) and start again; it continues where it stopped.' }
        }

        # 4e. NVIDIA GPU components (optional: a failure here only means processor mode)
        $cudaDone = $state['cuda']
        if ($needCuda) {
            Say 'Installing NVIDIA GPU components (about 1.3 GB, one time) ...' 'Cyan'
            $rc = Invoke-Pip -Py $Py -PipArgs (@('install', '--only-binary=:all:') + $PipSource + @('-r', $CudaFile)) -PipLog $pipLog
            if ($rc -eq 0) { $cudaDone = $cudaHash }
            else { Say 'The GPU components could not be installed. Continuing in processor mode; the next start will try again.' 'Yellow' }
        }

        # 4f. check that the engine really loads
        $v = Invoke-Native -Exe $Py -ArgList @('-c', 'import fastapi, uvicorn, numpy, ctranslate2, faster_whisper')
        Write-Log $v.Output
        if ($v.Code -ne 0) { Fail 'The app components did not install correctly.' 'Start again; if it repeats, send the logs folder to the person who shared the app.' }

        $mv = @{ lock = $lockHash }
        if ($cudaDone) { $mv['cuda'] = $cudaDone } else { $mv['cuda'] = 'none' }
        Write-Marker -Path $Marker -Values $mv
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
        Say 'Setup finished.' 'Green'
    }

    # ---- 5. start
    $ServerLog = Join-Path $Logs 'server.log'
    if (Test-Path -LiteralPath $ServerLog) { Move-Item -LiteralPath $ServerLog -Destination (Join-Path $Logs 'server.prev.log') -Force }
    $Url = 'http://localhost:' + $Port + '/'

    $env:CT2_DIR = $Model
    $state = Read-Marker -Path $Marker
    if (-not $useGpu -or $state['cuda'] -eq 'none') { $env:HW_DEVICE = 'cpu' }   # no GPU components: skip the GPU test
    $env:HF_HUB_OFFLINE = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONUTF8 = '1'

    Write-Host ''
    Say 'Starting. Your browser opens by itself in a moment (the first start can take a minute).' 'Green'
    Say ('If it does not, open ' + $Url + ' in Chrome or Edge.') 'White'
    Say 'Keep this window open while you use the app. Close it to stop.' 'Yellow'
    Write-Host ''

    $null = Start-Job -ScriptBlock {
        param($U)
        for ($i = 0; $i -lt 150; $i++) {
            try {
                $r = [System.Net.WebRequest]::Create($U)
                $r.Timeout = 1500
                $resp = $r.GetResponse()
                $resp.Close()
                if (-not $env:QT_NO_BROWSER) { Start-Process $U }   # QT_NO_BROWSER=1: test runs
                return
            } catch { Start-Sleep -Milliseconds 700 }
        }
    } -ArgumentList $Url

    Set-Location -LiteralPath $Back
    $ErrorActionPreference = 'Continue'
    & $Py -m uvicorn server:app --host 127.0.0.1 --port $Port 2>&1 | ForEach-Object {
        $line = "$_"
        Add-Content -LiteralPath $ServerLog -Value $line -Encoding UTF8
        if ($line -match 'ASR backend|test failed|Traceback|Error|error') { Write-Host ('  ' + $line) }
    }
    $rc = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'

    if ($rc -ne 0) {
        $tail = @(Get-Content -LiteralPath $ServerLog -Tail 40 -ErrorAction SilentlyContinue)
        if (@($tail -match '10048|address already in use|Errno 98').Count -gt 0) {
            Fail 'The network port is already in use.' 'Close other copies of this app (or restart the computer) and try again.'
        }
        Fail 'The app stopped unexpectedly.' 'Send the logs folder to the person who shared the app.'
    }
    Say 'Quran Tracker stopped.' 'White'
    exit 0
} catch {
    Write-Log ($_ | Out-String) 'ERROR'
    Fail ('Something unexpected went wrong: ' + $_.Exception.Message) 'Start again; if it repeats, send the logs folder to the person who shared the app.'
}
