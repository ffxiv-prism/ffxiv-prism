param([Parameter(Mandatory=$true)][string]$Manifest, [Parameter(Mandatory=$true)][string]$KeyPath, [Parameter(Mandatory=$true)][string]$PublicPath)
$ErrorActionPreference = 'Stop'
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
$json = '{"algorithm":"RS256","n":"' + $n + '","e":"' + $e + '"}'
[IO.File]::WriteAllText($PublicPath, $json, (New-Object Text.UTF8Encoding($false)))
$bytes = [IO.File]::ReadAllBytes($Manifest)
$signature = $rsa.SignData($bytes, [Security.Cryptography.CryptoConfig]::MapNameToOID('SHA256'))
[IO.File]::WriteAllText([IO.Path]::ChangeExtension($Manifest, '.sig'), [Convert]::ToBase64String($signature))
$rsa.Dispose()
