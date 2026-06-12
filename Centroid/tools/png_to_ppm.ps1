param(
    [Parameter(Mandatory = $true)]
    [string]$InputPngPath,

    [string]$OutputPpmPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

<# /**
 * Resolves the source PNG and default output PPM path.
 */ #>
$resolvedInputPngPath = (Resolve-Path -LiteralPath $InputPngPath).Path

<# /**
 * Keeps the converted file beside the PNG when no output path is provided.
 */ #>
if ($OutputPpmPath -eq "") {
    $inputDirectoryPath = Split-Path -Parent $resolvedInputPngPath
    $inputFileBaseName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedInputPngPath)
    $OutputPpmPath = Join-Path $inputDirectoryPath "$inputFileBaseName.ppm"
}

<# /**
 * Loads Windows built-in image support so no external converter is required.
 */ #>
Add-Type -AssemblyName System.Drawing

<# /**
 * Writes a binary P6 PPM file because the Centroid test runner already reads PPM.
 */ #>
$bitmapImage = [System.Drawing.Bitmap]::FromFile($resolvedInputPngPath)
try {
    $outputStream = [System.IO.File]::Open($OutputPpmPath, [System.IO.FileMode]::Create)
    try {
        $ppmHeader = "P6`n$($bitmapImage.Width) $($bitmapImage.Height)`n255`n"
        $ppmHeaderBytes = [System.Text.Encoding]::ASCII.GetBytes($ppmHeader)
        $outputStream.Write($ppmHeaderBytes, 0, $ppmHeaderBytes.Length)

        for ($pixelY = 0; $pixelY -lt $bitmapImage.Height; $pixelY++) {
            for ($pixelX = 0; $pixelX -lt $bitmapImage.Width; $pixelX++) {
                $pixelColor = $bitmapImage.GetPixel($pixelX, $pixelY)
                $outputStream.WriteByte($pixelColor.R)
                $outputStream.WriteByte($pixelColor.G)
                $outputStream.WriteByte($pixelColor.B)
            }
        }
    }
    finally {
        $outputStream.Close()
    }
}
finally {
    $bitmapImage.Dispose()
}

Write-Host "Wrote $OutputPpmPath"
