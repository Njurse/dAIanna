#include "pch.h"
#include <Windows.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#include "easyhook.h"
#include <string>
#include <ntstatus.h>
#pragma comment(lib, "OpenGL32.lib")
#pragma comment(lib, "EasyHook32.lib")

#define PIPE_NAME "\\\\.\\pipe\\daianna_pipe"

typedef void(APIENTRY* glDrawElements_t)(GLenum mode, GLsizei count, GLenum type, const void* indices);
glDrawElements_t original_glDrawElements = NULL;

void SendDataToPipe(const void* data, size_t size) {
    HANDLE hPipe = CreateFileA(PIPE_NAME, GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (hPipe != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(hPipe, data, static_cast<DWORD>(size), &written, NULL);
        char buffer[256];
        sprintf_s(buffer, "%lu bytes written\n", written);
        OutputDebugStringA(buffer);
        CloseHandle(hPipe);
    }
    else {
        OutputDebugStringA("Failed to open pipe\n");
    }
}

void APIENTRY Hooked_glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    size_t dataSize;
    switch (type) {
    case GL_UNSIGNED_BYTE:  dataSize = sizeof(GLubyte);  break;
    case GL_UNSIGNED_SHORT: dataSize = sizeof(GLushort); break;
    case GL_UNSIGNED_INT:   dataSize = sizeof(GLuint);   break;
    default: dataSize = 0; break;
    }
    if (dataSize > 0) {
        SendDataToPipe(indices, count * dataSize);
    }
    original_glDrawElements(mode, count, type, indices);
}

void HookOpenGLFunctions() {
    OutputDebugStringA("HookOpenGLFunctions called");
    HMODULE hModule = GetModuleHandle(L"opengl32.dll");
    if (hModule) {
        original_glDrawElements = (glDrawElements_t)GetProcAddress(hModule, "glDrawElements");
        if (original_glDrawElements) {
            NTSTATUS status = LhInstallHook(original_glDrawElements, Hooked_glDrawElements, NULL, NULL);
            if (status == STATUS_SUCCESS) {
                OutputDebugStringA("Hook installed successfully");
                ULONG ACLEntries[1] = { 0 };
                LhSetInclusiveACL(ACLEntries, 1, NULL);
            }
            else {
                char buffer[256];
                sprintf_s(buffer, "Failed to install hook: 0x%X\n", status);
                OutputDebugStringA(buffer);
            }
        }
        else {
            OutputDebugStringA("glDrawElements not found in this process");
        }
    }
    else {
        OutputDebugStringA("opengl32.dll not found");
    }
}

/*
extern "C" void __stdcall NativeInjectionEntryPoint(REMOTE_ENTRY_INFO* inRemoteInfo) {
    try {
        OutputDebugStringA("NativeInjectionEntryPoint called");
        HookOpenGLFunctions();
    }
    catch (const std::exception& e) {
        char buffer[256]; 
        sprintf_s(buffer, "Exception in NativeInjectionEntryPoint: %s", e.what());
        OutputDebugStringA(buffer);
    }
    catch (...) {
        OutputDebugStringA("Unknown exception in NativeInjectionEntryPoint");
    }
}
*/
BOOL WINAPI DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hModule);
        break;
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
