#ifndef MY_DEFINITIONS_H
#define MY_DEFINITIONS_H

// Define necessary types and structures
struct br_pixelmap {
    // Define the structure members based on actual application requirements
    int width;
    int height;
    // Add other members as needed
};

typedef struct {
    // Define tGraf_spec members based on actual requirements
    // Example:
    int width;
    int height;
    // Add other members as needed
} tGraf_spec;

typedef int tS32;
typedef unsigned int tU32;

// Declare external variables
extern int gASCII_table[128];
extern tU32 gKeyboard_bits[8];
extern int gASCII_shift_table[128];
extern tGraf_spec gGraf_specs[2];
extern char gNetwork_profile_fname[256];
extern tS32 gJoystick_min1y;
extern tS32 gJoystick_min2y;
extern tS32 gJoystick_min2x;
extern tS32 gRaw_joystick2y;
extern tS32 gRaw_joystick2x;
extern tS32 gRaw_joystick1y;
extern tS32 gRaw_joystick1x;
extern tS32 gJoystick_range2y;
extern tS32 gJoystick_range2x;
extern tS32 gJoystick_range1y;
extern tS32 gJoystick_range1x;
extern int gNo_voodoo;
extern int gSwitched_resolution;
extern int gReplay_override;
extern br_pixelmap* gReal_back_screen;
extern tS32 gJoystick_min1x;
extern br_pixelmap* gTemp_screen;
extern int gDOSGfx_initialized;
extern tU32 gUpper_loop_limit;
extern int gExtra_mem;
extern int gReal_back_screen_locked;
extern void (*gPrev_keyboard_handler)(void);

#endif // MY_DEFINITIONS_H
