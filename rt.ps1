<#
.SYNOPSIS
  Build, test, render, benchmark and profile the ray tracer.

.DESCRIPTION
  One entry point for the things you actually do, so none of them depend on
  remembering a working directory or a flag.

    .\rt.ps1 build                      configure + build Release
    .\rt.ps1 build -Config Debug        (never benchmark this)
    .\rt.ps1 test                       ctest: unit + golden image
    .\rt.ps1 test -Filter bvh           run one group of device tests
    .\rt.ps1 render cornell             render at the scene's defaults
    .\rt.ps1 render original -Width 400 -Spp 500 -Png
    .\rt.ps1 scenes                     list available scenes
    .\rt.ps1 bench                      timing matrix
    .\rt.ps1 bench -Quick
    .\rt.ps1 golden                     check the reference images
    .\rt.ps1 golden -Update             re-baseline them (review the diff!)
    .\rt.ps1 sanitize cornell           compute-sanitizer memcheck + leak check
    .\rt.ps1 profile cornell            Nsight Compute kernel profile
    .\rt.ps1 clean
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('build', 'test', 'render', 'scenes', 'bench', 'golden', 'sanitize', 'profile', 'clean', 'help')]
    [string]$Command = 'help',

    [Parameter(Position = 1)]
    [string]$Scene = 'original',

    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string]$Config = 'Release',

    [int]$Width, [int]$Height, [int]$Spp, [int]$Seed, [int]$MaxDepth,
    [string]$Out,
    [string]$Filter,
    [switch]$Png,       # convert the render to PNG afterwards
    [switch]$Quick,     # bench: small matrix
    [switch]$Update,    # golden: re-baseline
    [switch]$NoBuild    # skip the implicit build
)

$ErrorActionPreference = 'Stop'

# Captured at script scope on purpose. Inside a function, $PSBoundParameters
# refers to *that function's* parameters, so reading it from a helper silently
# reports nothing bound and every -Width/-Spp override gets dropped.
$ScriptParams = $PSBoundParameters

$Root = $PSScriptRoot
$BuildDir = Join-Path $Root 'build'
$ExeDir = Join-Path $BuildDir "bin\$Config"
$Exe = Join-Path $ExeDir 'rayTracer.exe'

function Fail($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# Run a native executable and check its exit code.
#
# The renderer deliberately writes progress and timing to stderr so that stdout
# stays a clean PPM pipe. PowerShell, with $ErrorActionPreference='Stop', turns
# any native stderr output into a terminating NativeCommandError -- so a
# perfectly successful render "fails" because it printed a progress bar. Drop to
# 'Continue' for the call and judge the result by $LASTEXITCODE, which is the
# only thing that actually reports success.
function Invoke-Native {
    param([string]$Path, [string[]]$Arguments, [string]$What = 'command')
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Path @Arguments } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit $LASTEXITCODE)" }
}

function Invoke-Build {
    Step "Configuring ($Config)"
    cmake -S $Root -B $BuildDir | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail 'cmake configure failed' }

    Step "Building ($Config)"
    cmake --build $BuildDir --config $Config --parallel
    if ($LASTEXITCODE -ne 0) { Fail 'build failed' }

    if ($Config -eq 'Debug') {
        Write-Host 'note: Debug compiles device code with -G. Correct, but not representative for timing.' -ForegroundColor Yellow
    }
}

function Assert-Exe {
    if (-not $NoBuild) { Invoke-Build }
    if (-not (Test-Path $Exe)) { Fail "rayTracer.exe not found at $Exe (run: .\rt.ps1 build)" }
}

# Translate the script's typed parameters into renderer flags, passing only
# those the user actually set so the scene's own defaults survive.
function Get-RenderArgs {
    $a = @('--scene', $Scene)
    if ($ScriptParams.ContainsKey('Width'))    { $a += @('--width', $Width) }
    if ($ScriptParams.ContainsKey('Height'))   { $a += @('--height', $Height) }
    if ($ScriptParams.ContainsKey('Spp'))      { $a += @('--spp', $Spp) }
    if ($ScriptParams.ContainsKey('Seed'))     { $a += @('--seed', $Seed) }
    if ($ScriptParams.ContainsKey('MaxDepth')) { $a += @('--max-depth', $MaxDepth) }
    return $a
}

switch ($Command) {

    'build' { Invoke-Build }

    'test' {
        Assert-Exe
        if ($Filter) {
            Step "Device tests matching '$Filter'"
            Invoke-Native (Join-Path $ExeDir 'test_device.exe') @($Filter) 'device tests'
            exit 0
        }
        Step 'Running tests'
        ctest --test-dir $BuildDir -C $Config --output-on-failure
        exit $LASTEXITCODE
    }

    'scenes' { Assert-Exe; Invoke-Native $Exe @('--list') 'scene list' }

    'render' {
        Assert-Exe
        $target = if ($Out) { $Out } else { "$Scene.ppm" }
        if (-not [System.IO.Path]::IsPathRooted($target)) {
            $target = Join-Path (Get-Location) $target
        }
        Step "Rendering '$Scene'"
        Invoke-Native $Exe (@(Get-RenderArgs) + @('--out', $target)) 'render'
        Write-Host "wrote $target"
        if ($Png) {
            Invoke-Native 'python' @((Join-Path $Root 'tools\ppm_to_png.py'), $target) 'ppm_to_png'
        }
    }

    'bench' {
        Assert-Exe
        $a = @((Join-Path $Root 'bench\run_bench.py'), '--exe', $Exe)
        if ($Quick) { $a += '--quick' }
        if ($ScriptParams.ContainsKey('Scene')) { $a += @('--scene', $Scene) }
        Invoke-Native 'python' $a 'benchmark'
    }

    'golden' {
        Assert-Exe
        $a = @((Join-Path $Root 'tests\run_golden.py'), '--exe', $Exe)
        if ($Update) { $a += '--update' }
        Invoke-Native 'python' $a 'golden-image check'
        if ($Update) {
            Write-Host 'Review before committing:' -ForegroundColor Yellow
            Write-Host '  python tools\ppm_to_png.py tests\golden\*.ppm --scale 3'
        }
    }

    'sanitize' {
        Assert-Exe
        if (-not (Get-Command compute-sanitizer -ErrorAction SilentlyContinue)) {
            Fail 'compute-sanitizer not on PATH (ships with the CUDA toolkit)'
        }
        # Small and cheap: memcheck costs roughly 10-50x, so a full-size render
        # would take all day and tell you nothing extra.
        Step "memcheck + leak-check on '$Scene'"
        Invoke-Native 'compute-sanitizer' @(
            '--tool', 'memcheck', '--leak-check', 'full', '--',
            $Exe, '--scene', $Scene, '--width', '64', '--height', '64',
            '--spp', '4', '--quiet', '--out', (Join-Path $ExeDir 'sanitize.ppm')) 'compute-sanitizer'
        Write-Host ''
        Write-Host 'Expected: 0 invalid accesses. Leaks are known (instancing wrappers do' -ForegroundColor DarkGray
        Write-Host 'not own their children; shared materials are deliberately non-owning).' -ForegroundColor DarkGray
    }

    'profile' {
        Assert-Exe
        if (-not (Get-Command ncu -ErrorAction SilentlyContinue)) {
            Fail 'ncu (Nsight Compute) not on PATH'
        }
        # The build is compiled with -lineinfo in optimised configs, so the
        # profiler can attribute counters back to source lines.
        Step "Profiling render_accumulate on '$Scene'"
        $a = Get-RenderArgs
        if (-not $ScriptParams.ContainsKey('Width'))  { $a += @('--width', 200) }
        if (-not $ScriptParams.ContainsKey('Height')) { $a += @('--height', 200) }
        if (-not $ScriptParams.ContainsKey('Spp'))    { $a += @('--spp', 16) }
        Invoke-Native 'ncu' (@(
            '--set', 'full', '--kernel-name', 'render_accumulate', '--launch-count', '1',
            '-o', (Join-Path $BuildDir "profile_$Scene"), '-f', '--', $Exe) +
            $a + @('--batch', '16', '--quiet', '--out', (Join-Path $ExeDir 'profile.ppm'))) 'ncu'
        Write-Host "report: $BuildDir\profile_$Scene.ncu-rep"
    }

    'clean' {
        Step 'Removing build/'
        if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
        Write-Host 'clean'
    }

    default { Get-Help $PSCommandPath -Detailed }
}
