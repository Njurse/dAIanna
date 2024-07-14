// dAIannaInjector.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "dAIannaInjector.h"
#include "Windows.h"
#include <easyhook.h>
#pragma comment(lib, "EasyHook32.lib")
#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Place code here.

    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_DAIANNAINJECTOR, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_DAIANNAINJECTOR));

    MSG msg;

    // Main message loop:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_DAIANNAINJECTOR));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_DAIANNAINJECTOR);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}


//Define the window controls - to do: cfg save carma directory so it will automatically inject without user input after it is discovered/specified by user
void CreateControls(HWND hWnd)
{
    // Get the size of the parent window (hWnd)
    RECT rect;
    GetClientRect(hWnd, &rect);
    int windowWidth = rect.right - rect.left;
    int windowHeight = rect.bottom - rect.top;

    // Calculate button dimensions
    int buttonWidth = 100;
    int buttonHeight = 30;

    // Calculate button position
    int buttonX = (windowWidth - buttonWidth) / 2;  // Center horizontally
    int buttonY = (windowHeight * 3) / 4;           // 3/4 down vertically

    // Create the button
    CreateWindowW(
        L"BUTTON",                       // Predefined class; Unicode assumed
        L"Inject DLL",                   // Button text
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles
        buttonX,                         // x position
        buttonY,                         // y position
        buttonWidth,                     // Button width
        buttonHeight,                    // Button height
        hWnd,                            // Parent window
        (HMENU)1001,                     // No menu.
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);                           // Pointer not needed.

    // Calculate help text dimensions and position
    int staticWidth = windowWidth - 25;  // Adjust as needed
    int staticHeight = 125;  // Adjust as needed
    int staticX = (windowWidth - staticWidth) / 2;  // Center horizontally
    int staticY = windowHeight / 2.5 - staticHeight / 2;  // 1/4 down vertically

    // Create the help text string
    CreateWindowW(
        L"STATIC",                       // Predefined class; Unicode assumed
        L"If you can read this you've cocked something up and dAIanna could not automatically perform hot DLL munging. Please check your configuration and restart dAIanna.py or attempt to manually inject the DLL below.",             // Text content
        WS_VISIBLE | WS_CHILD | SS_CENTER,           // Styles
        staticX,                         // x position
        staticY,                         // y position
        staticWidth,                     // Control width
        staticHeight,                    // Control height
        hWnd,                            // Parent window
        NULL,                            // No menu.
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);                           // Pointer not needed.



}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Store instance handle in our global variable

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, 320, 240, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }
   CreateControls(hWnd); // Create the button control
   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

void StartMunging(LPCWSTR exePath, LPCWSTR dllPath)
{
    // Set up the process creation parameters
    STARTUPINFO si = { sizeof(STARTUPINFO) };
    DWORD processId;

    // Call RhCreateAndInject
    /*/NTSTATUS mungingResult = RhCreateAndInject(
        (WCHAR*)exePath,              // Path to the target executable
        NULL,                         // Command line arguments (NULL if none)
        CREATE_DEFAULT_ERROR_MODE,    // Process creation flags
        EASYHOOK_INJECT_DEFAULT,      // Injection options
        (WCHAR*)dllPath,              // Path to the 32-bit DLL (NULL if not used)
        (WCHAR*)dllPath,              // Path to the 64-bit DLL (NULL if not used)
        NULL,                         // Pass any data to the injected DLL (NULL if none)
        0,                            // Size of the data (0 if none)
        &processId                    // Out process ID
    );
    */
    if (true)
    {
        // The string needs to be created and formatted before passing it to the MessageBox
        wchar_t message[512];
        swprintf(message, sizeof(message) / sizeof(wchar_t), L"CARMA95.exe path: %s\nDLL path: %s", exePath, dllPath);

        // Open a dialogue window with information
        MessageBox(NULL, message, L"Munging Successful", MB_OK | MB_ICONINFORMATION);
    }
    else
    {
        // Error attaching detour
        MessageBox(NULL, L"An error occurred while feverishly munging. What might that error be? The world may never know!", L"Munging Error", MB_OK | MB_ICONERROR);
        return;
    }
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
        {
            LPCWSTR carmaExePath = L"CARMA95.exe";
            LPCWSTR dAIannaHookPath = L"dAIannaHook.dll";
            int wmId = LOWORD(wParam);
            // Parse the menu selections:
            switch (wmId)
            {
            case 1001: //to do: find a belt and a stool for being lazy
                // Handle button click here
                StartMunging(carmaExePath, dAIannaHookPath);  // Call the function to perform the action
                break;
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code that uses hdc here...
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
