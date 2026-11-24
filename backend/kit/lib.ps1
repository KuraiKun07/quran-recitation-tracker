# lib.ps1 - helper functions for run.ps1. PowerShell 5.1 compatible. ASCII only.

function Write-Log {
    param([string]$Msg, [string]$Level = 'INFO')
    if (-not $script:LogFile) { return }
    try {
        $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Msg
        Add-Content -LiteralPath $script:LogFile -Value $line -Encoding UTF8
    } catch { }
}

function Say {
    param([string]$Msg, [string]$Color = 'White')
    Write-Host $Msg -ForegroundColor $Color
    Write-Log $Msg
}

# Lines look like: NVIDIA GeForce RTX 3060 Laptop GPU, 546.33, 6144   (csv, no header, no units)
function ConvertFrom-NvidiaSmi {
    param([string[]]$Lines)
    $res = @()
    foreach ($l in $Lines) {
        if ([string]::IsNullOrWhiteSpace($l)) { continue }
        $p = @($l.Split(',') | ForEach-Object { $_.Trim() })
        if ($p.Count -lt 3) { continue }
        $mem = 0
        if (-not [int]::TryParse($p[$p.Count - 1], [ref]$mem)) { continue }
        $name = ($p[0..($p.Count - 3)]) -join ', '
        $res += [pscustomobject]@{ Name = $name; Driver = $p[$p.Count - 2]; MemMiB = $mem }
    }
    return $res
}

function Get-NvidiaGpu {
    $cands = @((Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
               (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'))
    foreach ($c in $cands) {
        if (-not (Test-Path -LiteralPath $c)) { continue }
        try {
            $out = & $c '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader,nounits' 2>$null
            $g = @(ConvertFrom-NvidiaSmi -Lines @($out)) | Sort-Object MemMiB -Descending | Select-Object -First 1
            if ($g) { return $g }
        } catch { Write-Log ('nvidia-smi failed: ' + $_.Exception.Message) 'WARN' }
    }
    return $null
}

# CUDA 12 needs a driver of about 527.41 or newer on Windows. If this guess is too strict or too
# loose nothing breaks: the server tests the GPU itself at start and falls back to the processor.
function Test-GpuSuitable {
    param($Gpu, [string]$MinDriver = '527.41', [int]$MinMiB = 2000)
    if ($null -eq $Gpu) { return $false }
    $v = $null
    $min = $null
    if (-not [version]::TryParse([string]$Gpu.Driver, [ref]$v)) { return $false }
    if (-not [version]::TryParse($MinDriver, [ref]$min)) { return $false }
    return (($v -ge $min) -and ($Gpu.MemMiB -ge $MinMiB))
}

function Test-PathTooLong {
    param([string]$Path, [int]$Max = 100)
    return ($Path.Length -gt $Max)
}

function Get-FreeGB {
    param([string]$Path)
    try {
        $root = [System.IO.Path]::GetPathRoot($Path)
        if ([string]::IsNullOrEmpty($root) -or $root.StartsWith('\\')) { return $null }
        $d = New-Object System.IO.DriveInfo($root)
        return [math]::Round($d.AvailableFreeSpace / 1GB, 1)
    } catch { return $null }
}

function Test-Writable {
    param([string]$Dir)
    try {
        $f = Join-Path $Dir ('.w' + [guid]::NewGuid().ToString('N'))
        Set-Content -LiteralPath $f -Value 'x'
        Remove-Item -LiteralPath $f -Force
        return $true
    } catch { return $false }
}

function Get-MissingVcRuntime {
    $sys = Join-Path $env:SystemRoot 'System32'
    $missing = @()
    foreach ($n in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140_1.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $sys $n))) { $missing += $n }
    }
    return $missing
}

function Test-PortOpen {
    param([int]$Port)
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $c.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(500)) { $c.EndConnect($iar); return $true }
        return $false
    } catch { return $false } finally { $c.Close() }
}

function Test-HttpOk {
    param([string]$Url, [int]$TimeoutMs = 2000)
    try {
        $r = [System.Net.WebRequest]::Create($Url)
        $r.Timeout = $TimeoutMs
        $resp = $r.GetResponse()
        $ok = ([int]$resp.StatusCode -eq 200)
        $resp.Close()
        return $ok
    } catch { return $false }
}

function Get-File {
    param([string]$Url, [string]$Dest, [int]$Tries = 4)
    $ProgressPreference = 'SilentlyContinue'
    for ($i = 1; $i -le $Tries; $i++) {
        try {
            if (Test-Path -LiteralPath $Dest) { Remove-Item -LiteralPath $Dest -Force }
            Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing -TimeoutSec 120
            if ((Test-Path -LiteralPath $Dest) -and ((Get-Item -LiteralPath $Dest).Length -gt 0)) { return $true }
        } catch { Write-Log ('download try ' + $i + ' failed: ' + $Url + ' : ' + $_.Exception.Message) 'WARN' }
        if ($i -lt $Tries) { Start-Sleep -Seconds (2 * $i) }
    }
    return $false
}

# Run a native program, return exit code and combined text. Safe under $ErrorActionPreference = 'Stop'.
function Invoke-Native {
    param([string]$Exe, [string[]]$ArgList)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $rc = 1
    $out = @()
    try {
        $out = @(& $Exe @ArgList 2>&1 | ForEach-Object { "$_" })
        $rc = $LASTEXITCODE
    } finally { $ErrorActionPreference = $old }
    return [pscustomobject]@{ Code = $rc; Output = ($out -join "`n") }
}

# pip runs attached to the console so the user sees download progress; details go to PipLog.
# Start-Process is used on purpose: calling pip with & inside a function would return its output lines
# together with the exit code. Returns the exit code as an integer (1 if it cannot be read).
function Invoke-Pip {
    param([string]$Py, [string[]]$PipArgs, [string]$PipLog)
    $a = @('-m', 'pip') + $PipArgs + @('--disable-pip-version-check', '--no-warn-script-location',
                                       '--retries', '10', '--timeout', '60', '--log', $PipLog)
    $q = @($a | ForEach-Object { if ($_ -match '[\s"]') { '"' + $_ + '"' } else { $_ } })
    $p = Start-Process -FilePath $Py -ArgumentList ($q -join ' ') -NoNewWindow -Wait -PassThru
    $null = $p.Handle
    if ($null -eq $p.ExitCode) { return [int]1 }
    return [int]$p.ExitCode
}

# The embeddable Python ignores site-packages unless its ._pth file says so. Paths are relative to that file.
function Write-PthFile {
    param([string]$PyDir)
    $pth = Get-ChildItem -LiteralPath $PyDir -Filter 'python*._pth' | Select-Object -First 1
    if (-not $pth) { throw 'python._pth not found in the Python folder' }
    $lines = @(($pth.BaseName + '.zip'), '.', 'Lib\site-packages', '..\..\app\backend', 'import site')
    Set-Content -LiteralPath $pth.FullName -Value $lines -Encoding Ascii
}

function Read-Marker {
    param([string]$Path)
    $h = @{}
    if (Test-Path -LiteralPath $Path) {
        foreach ($l in (Get-Content -LiteralPath $Path)) {
            $i = $l.IndexOf('=')
            if ($i -gt 0) { $h[$l.Substring(0, $i)] = $l.Substring($i + 1) }
        }
    }
    return $h
}

function Write-Marker {
    param([string]$Path, [hashtable]$Values)
    $lines = @()
    foreach ($k in $Values.Keys) { $lines += ($k + '=' + $Values[$k]) }
    Set-Content -LiteralPath $Path -Value $lines -Encoding Ascii
}
