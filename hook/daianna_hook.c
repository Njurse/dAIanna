//dAIanna OpenGL hook
//in lieu of AI depth mapping i'm instead opting to intercept the vertex buffer and send it over stdio pipe to the dAIanna application for scene reconstruction
//This probably needs all kind of awful setup to compile properly but it's worth adding the source code to the repo
#include <Windows.h>
#include "EasyHook.h"
#include <GL/gl.h>
#include <iostream>
#include <fstream>

#define PIPE_NAME "\\\\.\\pipe\\daianna_pipe"

typedef void(APIENTRY* glDrawElements_t)(GLenum mode, GLsizei count, GLenum type, const void* indices);
glDrawElements_t original_glDrawElements = NULL;

void SendDataToPipe(const void* data, size_t size) {
    HANDLE hPipe = CreateFileA(
        PIPE_NAME,          // pipe name
        GENERIC_WRITE,      // write access
        0,                  // no sharing
        NULL,               // default security attributes
        OPEN_EXISTING,      // opens existing pipe
        0,                  // default attributes
        NULL);              // no template file

    if (hPipe != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(hPipe, data, size, &written, NULL);
        CloseHandle(hPipe);
    } else {
        // Handle error if needed
    }
}

void APIENTRY Hooked_glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    // Capture vertex data here
    // For example, we can send the indices data to the pipe
    SendDataToPipe(indices, count * sizeof(type));  // Adjust size as necessary

    // Call the original function
    original_glDrawElements(mode, count, type, indices);
}

void HookOpenGLFunctions() {
    // Initialize EasyHook
    if (FAILED(RhInitialize())) {
        // Handle initialization failure
    }

    // Install the hook for glDrawElements
    HMODULE hOpenGL = GetModuleHandleA("opengl32.dll");
    original_glDrawElements = (glDrawElements_t)GetProcAddress(hOpenGL, "glDrawElements");

    if (FAILED(RhCreateAndInstallHook(
        (PVOID)original_glDrawElements,
        Hooked_glDrawElements,
        NULL,
        (TRACED_HOOK_HANDLE*)&original_glDrawElements))) {
        // Handle hook installation failure
    }

    // Enable the hook
    RhEnableHook((PVOID)original_glDrawElements);
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        HookOpenGLFunctions();
        break;
    case DLL_PROCESS_DETACH:
        // Cleanup
        RhRemoveHooks();
        break;
    }
    return TRUE;
}
