param([Parameter(Mandatory=$true)][string]$Manifest, [Parameter(Mandatory=$true)][string]$KeyPath, [Parameter(Mandatory=$true)][string]$PublicPath, [switch]$InitializeKey, [switch]$RotateKey)
$ErrorActionPreference = 'Stop'
$existing = $null
if (Test-Path -LiteralPath $PublicPath) {
    $existing = [IO.File]::ReadAllText($PublicPath) | ConvertFrom-Json
}
$creating = -not (Test-Path -LiteralPath $KeyPath)
if ($creating -and -not $InitializeKey) {
    throw 'Signing key missing. Restore the private key, or explicitly initialize a replacement key.'
}
if ($creating -and $existing -and -not $RotateKey) {
    throw 'A trusted public key already exists. Use -InitializeKey -RotateKey to retain it and create a replacement.'
}
if ($RotateKey -and -not $creating) {
    throw 'Refusing to overwrite an existing private key.'
}
$rsa = New-Object System.Security.Cryptography.RSACryptoServiceProvider(3072)
$rsa.PersistKeyInCsp = $false
if (Test-Path -LiteralPath $KeyPath) {
    $rsa.FromXmlString([IO.File]::ReadAllText($KeyPath))
} else {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($KeyPath)) | Out-Null
    [IO.File]::WriteAllText($KeyPath, $rsa.ToXmlString($true))
}
$pub = $rsa.ExportParameters($false)
$n = ([BitConverter]::ToString($pub.Modulus)).Replace('-', '').ToLower()
$e = ([BitConverter]::ToString($pub.Exponent)).Replace('-', '').ToLower()
if (-not $creating -and $existing -and ($existing.n -ne $n -or $existing.e -ne $e)) {
    throw 'Private key does not match the current public key. Refusing to change release trust.'
}
$public = [ordered]@{algorithm='RS256'; n=$n; e=$e}
if ($creating -and $existing) {
    $public.previous_keys = @(@{algorithm='RS256'; n=$existing.n; e=$existing.e}) + @($existing.previous_keys | Where-Object { $null -ne $_ })
} elseif ($existing -and $existing.previous_keys) {
    $public.previous_keys = @($existing.previous_keys)
}
$json = $public | ConvertTo-Json -Depth 5 -Compress
[IO.File]::WriteAllText($PublicPath, $json, (New-Object Text.UTF8Encoding($false)))
$bytes = [IO.File]::ReadAllBytes($Manifest)
$signature = $rsa.SignData($bytes, [Security.Cryptography.CryptoConfig]::MapNameToOID('SHA256'))
[IO.File]::WriteAllText([IO.Path]::ChangeExtension($Manifest, '.sig'), [Convert]::ToBase64String($signature))
$rsa.Dispose()
