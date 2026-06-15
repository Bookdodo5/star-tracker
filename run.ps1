<#
.SYNOPSIS
    Single entry point for the Star Tracker pipeline.

.DESCRIPTION
    Wraps the build, benchmark, and identification workflows so the whole project
    is driven from one script. Subcommands:

        build                         Configure and compile every identifier and centroid target.
        test                          Run the unit test and the synthetic benchmark summary.
        identify <image> [fov]        Run the full pipeline (PNG/PPM -> centroids -> attitude).
        fetch --ra X --dec Y --fov Z  Download a DSS image for the field and identify it.

.EXAMPLE
    .\run.ps1 build
    .\run.ps1 test
    .\run.ps1 identify .\centroid\test-image\10h16m56s-59-51-22.png 10
    .\run.ps1 fetch --ra 83.8 --dec -5.4 --fov 10
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("build", "test", "identify", "fetch")]
    [string]$Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$IdentifierBuildDir = Join-Path $ProjectRoot "identifier\build-generated-release"
$CentroidBuildDir = Join-Path $ProjectRoot "centroid\build-mingw"
$DemoExe = Join-Path $IdentifierBuildDir "demo_centroid_compare.exe"
$BatchExe = Join-Path $IdentifierBuildDir "batch_synthetic_compare.exe"
$UnitTestExe = Join-Path $IdentifierBuildDir "test_star_identifier.exe"
$CentroidExe = Join-Path $CentroidBuildDir "centroid_extract.exe"

<# /**
 * Configures and builds all identifier and centroid targets.
 */ #>
function Invoke-Build {
    Write-Host "[build] Configuring identifier (Release)..."
    cmake -S (Join-Path $ProjectRoot "identifier") -B $IdentifierBuildDir -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "identifier configure failed" }
    cmake --build $IdentifierBuildDir --target demo_centroid_compare batch_synthetic_compare test_star_identifier
    if ($LASTEXITCODE -ne 0) { throw "identifier build failed" }

    Write-Host "[build] Configuring centroid pipeline..."
    cmake -S (Join-Path $ProjectRoot "centroid") -B $CentroidBuildDir -G "MinGW Makefiles"
    if ($LASTEXITCODE -ne 0) { throw "centroid configure failed" }
    cmake --build $CentroidBuildDir --target centroid_extract
    if ($LASTEXITCODE -ne 0) { throw "centroid build failed" }

    Write-Host "[build] Done."
}

<# /**
 * Runs the unit test and the synthetic benchmark, printing the summary table.
 */ #>
function Invoke-Test {
    if (-not (Test-Path $UnitTestExe)) { throw "Missing $UnitTestExe - run '.\run.ps1 build' first." }
    Write-Host "[test] Unit test:"
    & $UnitTestExe
    if ($LASTEXITCODE -ne 0) { throw "Unit test failed" }

    $samples = if ($Rest.Count -ge 1) { $Rest[0] } else { "100" }
    $fov = if ($Rest.Count -ge 2) { $Rest[1] } else { "10" }
    Write-Host ""
    Write-Host "[test] Synthetic benchmark ($samples samples, FOV=$fov):"
    & $BatchExe $samples $fov 10
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed" }
}

<# /**
 * Reads the pixel width and height from a binary P6 PPM header.
 */ #>
function Get-PpmSize {
    param([string]$PpmPath)
    $stream = [System.IO.File]::OpenRead($PpmPath)
    try {
        $reader = New-Object System.IO.StreamReader($stream)
        $tokens = New-Object System.Collections.Generic.List[string]
        $current = ""
        while ($tokens.Count -lt 3 -and -not $reader.EndOfStream) {
            $ch = [char]$reader.Read()
            if ($ch -match '\s') {
                if ($current.Length -gt 0) { $tokens.Add($current); $current = "" }
            } else {
                $current += $ch
            }
        }
        # tokens: [0]=magic(P6) [1]=width [2]=height
        return [pscustomobject]@{ Width = [int]$tokens[1]; Height = [int]$tokens[2] }
    } finally {
        $stream.Close()
    }
}

<# /**
 * Runs the full identification pipeline on a PNG or PPM image.
 */ #>
function Invoke-Identify {
    param([string]$ImagePath, [string]$Fov = "10")
    if (-not (Test-Path $ImagePath)) { throw "Image not found: $ImagePath" }
    if (-not (Test-Path $CentroidExe)) { throw "Missing $CentroidExe - run '.\run.ps1 build' first." }

    $resolved = (Resolve-Path -LiteralPath $ImagePath).Path
    $extension = [System.IO.Path]::GetExtension($resolved).ToLowerInvariant()
    if ($extension -eq ".png") {
        Write-Host "[identify] Converting PNG to PPM..."
        & (Join-Path $ProjectRoot "centroid\tools\png_to_ppm.ps1") $resolved
        $ppmPath = [System.IO.Path]::ChangeExtension($resolved, ".ppm")
    } elseif ($extension -eq ".ppm") {
        $ppmPath = $resolved
    } else {
        throw "Unsupported image type '$extension' (expected .png or .ppm)"
    }

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($ppmPath)
    $outputsDir = Join-Path $ProjectRoot "outputs"
    New-Item -ItemType Directory -Force -Path $outputsDir | Out-Null
    $starsCsv = Join-Path $outputsDir "$baseName`_stars.csv"

    Write-Host "[identify] Extracting centroids..."
    & $CentroidExe $ppmPath $starsCsv
    if ($LASTEXITCODE -ne 0) { throw "Centroid extraction failed" }

    $size = Get-PpmSize -PpmPath $ppmPath
    Write-Host "[identify] Identifying ($($size.Width)x$($size.Height), FOV=$Fov)..."
    & $DemoExe $starsCsv $size.Width $size.Height $Fov
    if ($LASTEXITCODE -ne 0) { throw "Identification failed" }
}

<# /**
 * Parses --ra/--dec/--fov/--size flags from the remaining arguments.
 */ #>
function Get-FetchArgs {
    $parsed = @{ ra = $null; dec = $null; fov = "10"; size = "877" }
    for ($i = 0; $i -lt $Rest.Count - 1; $i++) {
        switch ($Rest[$i]) {
            "--ra"   { $parsed.ra = $Rest[$i + 1]; $i++ }
            "--dec"  { $parsed.dec = $Rest[$i + 1]; $i++ }
            "--fov"  { $parsed.fov = $Rest[$i + 1]; $i++ }
            "--size" { $parsed.size = $Rest[$i + 1]; $i++ }
        }
    }
    if ($null -eq $parsed.ra -or $null -eq $parsed.dec) {
        throw "fetch requires --ra and --dec (e.g. .\run.ps1 fetch --ra 83.8 --dec -5.4 --fov 10)"
    }
    return $parsed
}

<# /**
 * Fetches a DSS image for a sky field and runs identification on it.
 */ #>
function Invoke-Fetch {
    $args = Get-FetchArgs
    $outputsDir = Join-Path $ProjectRoot "outputs"
    New-Item -ItemType Directory -Force -Path $outputsDir | Out-Null
    $ppmPath = Join-Path $outputsDir "fetch_dss.ppm"

    Write-Host "[fetch] Downloading DSS image RA=$($args.ra) DEC=$($args.dec) FOV=$($args.fov)..."
    python (Join-Path $ProjectRoot "scripts\fetch_dss_image.py") `
        --ra $args.ra --dec $args.dec --fov $args.fov --size $args.size --output $ppmPath
    if ($LASTEXITCODE -ne 0) { throw "DSS fetch failed" }

    Invoke-Identify -ImagePath $ppmPath -Fov $args.fov
}

switch ($Command) {
    "build"    { Invoke-Build }
    "test"     { Invoke-Test }
    "identify" {
        if ($Rest.Count -lt 1) { throw "Usage: .\run.ps1 identify <image.png|.ppm> [fov]" }
        $fov = if ($Rest.Count -ge 2) { $Rest[1] } else { "10" }
        Invoke-Identify -ImagePath $Rest[0] -Fov $fov
    }
    "fetch"    { Invoke-Fetch }
}
