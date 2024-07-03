import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import depth_module
import daianna_intro

def grab_window(window_title):
    try:
        # Find the window by title
        window = gw.getWindowsWithTitle(window_title)[0]
        
        # Activate the window (focus)
        #window.activate()

        # Get the coordinates of the window
        left, top, width, height = window.left, window.top, window.width, window.height

        # Capture screenshot of the window
        screen = pyautogui.screenshot(region=(left, top, 642, 507))
        screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

        return screen

    except IndexError:
        raise Exception(f"Window not found: {window_title}")
        
# Function to check if HUD is detected
def is_hud_detected(screen):
    # Load the template image (HUD screenshot)
    template = cv2.imread('hud_template.png', cv2.IMREAD_COLOR)
    
    # Convert images to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Threshold for match detection (adjust as needed)
    threshold = 0.6
    #print(f"HUD similarity: {cv2.minMaxLoc(result)}")    
    # Check if match is found
    if max_val >= threshold:
        return True
    else:
        return False      
def create_hud_mask(screen):
    # Load the template image (HUD screenshot)
    template = cv2.imread('hud_template.png', cv2.IMREAD_COLOR)
    
    # Convert images to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Threshold for match detection (adjust as needed)
    threshold = 0.60
    #print(f"HUD similarity: {cv2.minMaxLoc(result)}")
    
    # Create a mask based on the matching result
    mask = np.ones_like(screen_gray, dtype=np.uint8)
    
    if max_val >= threshold:
        # Get the location of the HUD
        hud_top_left = max_loc
        hud_bottom_right = (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0])
        
        # Set the HUD area in the mask to 0
        mask[hud_top_left[1]:hud_bottom_right[1], hud_top_left[0]:hud_bottom_right[0]] = 0
    
    return mask

def grab_game_screenshot_nohud(screen):
    # Create the HUD mask
    mask = create_hud_mask(screen)
    
    # Apply the mask to the screenshot to avoid capturing the HUD
    no_hud_screen = cv2.bitwise_and(screen, screen, mask=mask)
    
    return no_hud_screen        

def is_main_menu_detected(screen):
    # Load the template image (main menu screenshot)
    template = cv2.imread('menu_template.png', cv2.IMREAD_COLOR)
    
    # Convert images to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Threshold for match detection (adjust as needed)
    threshold = 0.8
    
    # Check if match is found
    if max_val >= threshold:
        return True
    else:
        return False

def preprocess_screen(screen, template):
    # Extract alpha channel from template image
    alpha_channel = template[:, :, 3]

    # Create mask from alpha channel
    mask = alpha_channel > 0  # Mask will be True where alpha > 0 (non-transparent)

    # Apply mask to screen capture
    masked_screen = np.zeros_like(screen, dtype=np.uint8)
    masked_screen[mask] = screen[mask]

    return masked_screen

def main():
    window_title = "Carmageddon"
    print("Initializing main thread...") #to do: switch to proper logging, absolutely necessary if i decide async is required
    try:
        while True:
            # Capture from the specific window
            screen = grab_window(window_title)
            if screen is None:
                print("Issue capturing the window, skipping this generation.")
                continue  # Skip this iteration if screen capture failed
            else:
                # Check if HUD is detected (preprocess only if not in main menu)
                if not is_main_menu_detected(screen):
                    hud_template = cv2.imread('hud_template.png', cv2.IMREAD_UNCHANGED)
                    masked_screen = preprocess_screen(screen, hud_template)
                    if is_hud_detected(masked_screen):
                        #print("In-game HUD detected!")
                        no_hud_screen = grab_game_screenshot_nohud(screen)
                        color_screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
                        depth_module.process_depth_mapping(no_hud_screen)
                    else:
                        continue
                        #print("Not in-game and no menu detected.")
                else:
                    continue
                    #print("Main menu detected!")

                # Display the captured screen
                cv2.imshow('Game Screen', screen)
                cv2.waitKey(1)  # Wait for 1 millisecond (non-blocking)
            
            # Check for 'q' key to exit loop and close window
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
            
            # Wait for 0.25 seconds (quarter second)
            time.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    daianna_intro.play_intro()
    depth_module.initialize_midas()
    depth_module.process_video("sample_input.mkv")
    #main() is commented out at the moment because we can test mapping from a video then when everything is correct and tracing as it should the video can be swapped for opencv capture
