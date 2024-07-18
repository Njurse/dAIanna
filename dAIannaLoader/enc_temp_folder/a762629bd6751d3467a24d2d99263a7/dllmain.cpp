#include "pch.h"
#include <Windows.h>
#include <ddraw.h>
#include "easyhook.h"
#include <string>
#include <iostream>
#include <ntstatus.h>

#pragma comment(lib, "EasyHook32.lib")

// Function pointer types for the DirectDraw functions
typedef HRESULT(WINAPI* DirectDrawCreate_t)(GUID*, LPDIRECTDRAW*, IUnknown*);
typedef HRESULT(WINAPI* DirectDrawCreateEx_t)(GUID*, LPVOID*, REFIID, IUnknown*);
typedef HRESULT(WINAPI* CreateSurface_t)(LPDIRECTDRAW, LPDDSURFACEDESC, LPDIRECTDRAWSURFACE*, IUnknown*);
typedef HRESULT(WINAPI* Blt_t)(LPDIRECTDRAWSURFACE, LPRECT, LPDIRECTDRAWSURFACE, LPRECT, DWORD, LPDDBLTFX);

// Original function pointers
DirectDrawCreate_t OriginalDirectDrawCreate = NULL;
DirectDrawCreateEx_t OriginalDirectDrawCreateEx = NULL;
CreateSurface_t OriginalCreateSurface = NULL;
Blt_t OriginalBlt = NULL;

// Hooked functions
HRESULT WINAPI HookedDirectDrawCreate(GUID* lpGUID, LPDIRECTDRAW* lplpDD, IUnknown* pUnkOuter) {
    MessageBox(NULL, L"DirectDrawCreate called", L"Hooked Function", MB_OK | MB_ICONINFORMATION);
    return OriginalDirectDrawCreate(lpGUID, lplpDD, pUnkOuter);
}

HRESULT WINAPI HookedDirectDrawCreateEx(GUID* lpGUID, LPVOID* lplpDD, REFIID iid, IUnknown* pUnkOuter) {
    MessageBox(NULL, L"DirectDrawCreateEx called", L"Hooked Function", MB_OK | MB_ICONINFORMATION);
    return OriginalDirectDrawCreateEx(lpGUID, lplpDD, iid, pUnkOuter);
}

HRESULT WINAPI HookedCreateSurface(LPDIRECTDRAW lpDD, LPDDSURFACEDESC lpDDSurfaceDesc, LPDIRECTDRAWSURFACE* lplpDDSurface, IUnknown* pUnkOuter) {
    MessageBox(NULL, L"CreateSurface called", L"Hooked Function", MB_OK | MB_ICONINFORMATION);
    return OriginalCreateSurface(lpDD, lpDDSurfaceDesc, lplpDDSurface, pUnkOuter);
}

HRESULT WINAPI HookedBlt(LPDIRECTDRAWSURFACE lpDDSurfaceDest, LPRECT lpDestRect, LPDIRECTDRAWSURFACE lpDDSurfaceSrc, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx) {
    MessageBox(NULL, L"Blt called", L"Hooked Function", MB_OK | MB_ICONINFORMATION);
    return OriginalBlt(lpDDSurfaceDest, lpDestRect, lpDDSurfaceSrc, lpSrcRect, dwFlags, lpDDBltFx);
}

// Function to install hooks
void InstallHooks() {
    HMODULE hModule = LoadLibrary(L"ddraw.dll");
    if (hModule == NULL) {
        MessageBox(NULL, L"Failed to load ddraw.dll!", L"Error", MB_OK | MB_ICONERROR);
        return;
    }

    /*/ Hook DirectDrawCreate
    OriginalDirectDrawCreate = reinterpret_cast<DirectDrawCreate_t>(GetProcAddress(hModule, "DirectDrawCreate"));
    if (OriginalDirectDrawCreate && FAILED(LhInstallHook(
        OriginalDirectDrawCreate,
        HookedDirectDrawCreate,
        NULL,
        reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalDirectDrawCreate)
    ))) {
        MessageBox(NULL, L"Failed to hook DirectDrawCreate!", L"Error", MB_OK | MB_ICONERROR);
    }
    */
    // Hook DirectDrawCreateEx
    OriginalDirectDrawCreateEx = reinterpret_cast<DirectDrawCreateEx_t>(GetProcAddress(hModule, "DirectDrawCreateEx"));
    if (OriginalDirectDrawCreateEx && FAILED(LhInstallHook(
        OriginalDirectDrawCreateEx,
        HookedDirectDrawCreateEx,
        NULL,
        reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalDirectDrawCreateEx)
    ))) {
        MessageBox(NULL, L"Failed to hook DirectDrawCreateEx!", L"Error", MB_OK | MB_ICONERROR);
    }

    // Hook CreateSurface
    OriginalCreateSurface = reinterpret_cast<CreateSurface_t>(GetProcAddress(hModule, "CreateSurface"));
    if (OriginalCreateSurface)
    {
        if(FAILED(LhInstallHook(
            OriginalCreateSurface,
            HookedCreateSurface,
            NULL,
            reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalCreateSurface)
        ))) {
            MessageBox(NULL, L"Failed to hook CreateSurface!", L"Error", MB_OK | MB_ICONERROR);
        }
        else {
            MessageBox(NULL, L"CreateSurface!", L"Function Hooked", MB_OK | MB_ICONERROR);
        }
    }

    // Hook Blt
    OriginalBlt = reinterpret_cast<Blt_t>(GetProcAddress(hModule, "Blt"));
    if (OriginalBlt && FAILED(LhInstallHook(
        OriginalBlt,
        HookedBlt,
        NULL,
        reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalBlt)
    ))) {
        MessageBox(NULL, L"Failed to hook Blt!", L"Error", MB_OK | MB_ICONERROR);
    }
    if (OriginalBlt)
    {
        MessageBox(NULL, L"Fail32523523235ed to hook DirectDrawCreateEx!", L"Error", MB_OK | MB_ICONERROR);
    }

    // Activate hooks for all threads
    ULONG ACLEntries[1] = { 0 };
    LhSetInclusiveACL(ACLEntries, 1, reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalDirectDrawCreate));
    LhSetInclusiveACL(ACLEntries, 1, reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalDirectDrawCreateEx));
    LhSetInclusiveACL(ACLEntries, 1, reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalCreateSurface));
    LhSetInclusiveACL(ACLEntries, 1, reinterpret_cast<HOOK_TRACE_INFO*>(&OriginalBlt));
}

// Entry point for the injected DLL
extern "C" void __declspec(dllexport) __stdcall NativeInjectionEntryPoint(REMOTE_ENTRY_INFO* inRemoteInfo) {
    InstallHooks();
    MessageBox(
        NULL,
        L"dAIannaHook.dll successfully injected inside of CARMA95.exe",
        L"dAIannaHook is now running!",
        MB_OK | MB_ICONINFORMATION
    );
}