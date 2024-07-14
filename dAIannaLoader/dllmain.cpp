#include <Windows.h>
#include "pch.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <easyhook.h>
#include <iostream>
#include <fstream>
#pragma comment(lib, "EasyHook32.lib")

//Current plan is injecting dAIannaHook.dll into CARMA95.exe to extract game information and relay it over an io pipe to dAIanna.py
#define PIPE_NAME "\\\\.\\pipe\\daianna_pipe"

typedef void(APIENTRY* glDrawElements_t)(GLenum mode, GLsizei count, GLenum type, const void* indices);
glDrawElements_t original_glDrawElements = NULL;

//Draw to the screen at given coordinates (draw functionality implementation if for whatever reason it is needed besides debugging)
void RenderIndicator(float x, float y, float size) {
    // Example: Render a colored triangle
    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f); // Red color
    glVertex2f(x, y);
    glVertex2f(x + size, y);
    glVertex2f(x + size / 2, y + size);
    glEnd();
}

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
        printf("%lu bytes written\n", written);
        CloseHandle(hPipe);
    }
    else {
        // Handle error if needed
    }
}

void APIENTRY Hooked_glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    // Capture vertex data here
    // For example, we can send the indices data to the pipe
    SendDataToPipe(indices, count * sizeof(type));  // Adjust size as necessary
    RenderIndicator(64.0f, 64.0f, 32.0f);
    // Call the original function
    original_glDrawElements(mode, count, type, indices);
}

void HookOpenGLFunctions() {

    return;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        HookOpenGLFunctions();
        break;
    case DLL_PROCESS_DETACH:
        // Cleanup
        break;
    }
    return TRUE;
}
