#include "pch.h"
#include <Windows.h>
#include <ddraw.h> // DirectDraw header
#include "easyhook.h"
#include <iostream>
#include <fstream>

// Original DirectDraw function prototype
typedef HRESULT(WINAPI* Blt_t)(LPDIRECTDRAWSURFACE lpDDSurfaceDest, LPRECT lpDestRect, LPDIRECTDRAWSURFACE lpDDSurfaceSrc, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx);

// Original DirectDraw function prototypes
typedef HRESULT(WINAPI* Blt_t)(LPDIRECTDRAWSURFACE, LPRECT, LPDIRECTDRAWSURFACE, LPRECT, DWORD, LPDDBLTFX);
typedef HRESULT(WINAPI* Flip_t)(LPDIRECTDRAWSURFACE, LPDIRECTDRAWSURFACE, DWORD);

// Pointers to the original functions
Blt_t OriginalBlt = NULL;
Flip_t OriginalFlip = NULL;


// Hook function for Blt
HRESULT WINAPI HookedBlt(LPDIRECTDRAWSURFACE lpDDSurfaceDest, LPRECT lpDestRect, LPDIRECTDRAWSURFACE lpDDSurfaceSrc, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx) {
    // Log the data to a file or perform other actions
    std::ofstream logFile("ddraw_hook_log.txt", std::ios::out | std::ios::app);
    if (logFile.is_open()) {
        logFile << "Hooked Blt function called." << std::endl;
        // Optionally log parameters or other relevant information
        logFile.close();
    }
    else {
        std::cerr << "Failed to open log file for writing!" << std::endl;
    }

    // Call the original function
    if (OriginalBlt) {
        return OriginalBlt(lpDDSurfaceDest, lpDestRect, lpDDSurfaceSrc, lpSrcRect, dwFlags, lpDDBltFx);
    }
    else {
        return E_FAIL; // Handle error if OriginalBlt is not valid
    }
}

// Function to install the hook
extern "C" void __declspec(dllexport) __stdcall NativeInjectionEntryPoint(REMOTE_ENTRY_INFO* inRemoteInfo);
void __stdcall NativeInjectionEntryPoint(REMOTE_ENTRY_INFO* inRemoteInfo) {
    // Initialize EasyHook
    HMODULE hModule = LoadLibrary(L"ddraw.dll"); // Replace with actual module name
    if (hModule == NULL) {
        MessageBox(NULL, L"Failed to get module handle!", L"Error", MB_OK | MB_ICONERROR);
        return;
    }

    // Install hook for Blt function
    if (FAILED(LhInstallHook(
        GetProcAddress(hModule, "Blt"),
        HookedBlt,
        NULL,
        reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalBlt)
    ))) {
        MessageBox(NULL, L"Failed to install hook for Blt function!", L"Error", MB_OK | MB_ICONERROR);
        return;
    }

    // Enable the hook
    HOOK_TRACE_INFO hookInfo = { NULL }; // Initialize hook info structure
    ULONG aclEntries[1] = { 0 }; // Example ACL with one entry
    if (FAILED(LhSetExclusiveACL(aclEntries, 1, &hookInfo))) {
        MessageBox(NULL, L"Failed to enable hook!", L"Error", MB_OK | MB_ICONERROR);
        exit(1);
            
    }

    MessageBox(
        NULL,
        L"dAIannaHook.dll successfully injected inside of dethrace.exe",
        L"dAIannaHook is now running!",
        MB_OK | MB_ICONWARNING
    );
} 