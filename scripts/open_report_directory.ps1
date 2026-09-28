param([Parameter(Mandatory = $true)][string]$Directory)

$ErrorActionPreference = 'Stop'
$targetDirectory = (Get-Item -LiteralPath $Directory).FullName.TrimEnd('\')

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class PrismExplorerWindow {
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hwnd, int command);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(
        IntPtr hwnd, IntPtr after, int x, int y, int width, int height, uint flags);

    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, IntPtr processId);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint first, uint second, bool attach);

    public static bool Raise(IntPtr hwnd) {
        if (IsIconic(hwnd)) ShowWindowAsync(hwnd, 9); // SW_RESTORE
        // Raise only the requested folder, then restore normal z-order.
        // Explorer must not stay permanently above other applications.
        const uint flags = 0x0001 | 0x0002 | 0x0040; // NOSIZE | NOMOVE | SHOWWINDOW
        uint currentThread = GetCurrentThreadId();
        uint foregroundThread = GetWindowThreadProcessId(GetForegroundWindow(), IntPtr.Zero);
        bool attached = false;
        try {
            // This action originates from an explicit local button click.
            // Temporarily share foreground input so the hidden helper can
            // activate Explorer, and always release it before returning.
            if (foregroundThread != 0 && foregroundThread != currentThread)
                attached = AttachThreadInput(currentThread, foregroundThread, true);
            SetWindowPos(hwnd, new IntPtr(-1), 0, 0, 0, 0, flags);
            SetForegroundWindow(hwnd);
        } finally {
            SetWindowPos(hwnd, new IntPtr(-2), 0, 0, 0, 0, flags);
            if (attached) AttachThreadInput(currentThread, foregroundThread, false);
        }
        return GetForegroundWindow() == hwnd;
    }
}
"@

$shell = New-Object -ComObject Shell.Application
try {
    # Reuse an existing folder window before asking Explorer to open one.
    $opened = $false
    $deadline = [DateTime]::UtcNow.AddSeconds(8)
    do {
        foreach ($window in $shell.Windows()) {
            try {
                if ([IO.Path]::GetFileName($window.FullName) -ine 'explorer.exe') { continue }
                $windowPath = ([string]$window.Document.Folder.Self.Path).TrimEnd('\')
                if ($windowPath -ine $targetDirectory) { continue }
                if ([PrismExplorerWindow]::Raise([IntPtr][long]$window.HWND)) { exit 0 }
            } catch {
                # Explorer can close/navigate a window during enumeration.
                continue
            }
        }
        if (-not $opened) {
            # Open the folder normally; Explore requests the navigation tree view.
            $shell.Open($targetDirectory)
            $opened = $true
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'The report folder window did not become available.'
} finally {
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
}
