# dAIanna - A Reinforcement Learning Framework for Carmageddon Max Pack

_dAIanna_ is an ambitious project aimed at enhancing the gaming experience in Carmageddon Max Pack through advanced AI techniques. The current testing environment is Windows 10 with AMD ROCm in lieu of CUDA

## dAIannaHook.dll
Initially, _dAIanna_ was planned to utilize the MiDaS neural network to produce a depth map from a screen capture of the game, and a perspective correction algorithm would attempt to correct the depth map to more accurately reflect the in-game environment. However, the massive overhead from MiDaS and iterating over tens of thousands of points resulted in long processing times (Well before any game-playing AI code gets written, the information gathering stage of each update needs to be fast enough to leave as much time as possible for the AI to process its surroundings.)  
To solve this, the MiDaS system of information gathering is being superseded by a dynamic-link library (DLL) aptly named _dAIannaHook.dll_. This DLL will intercept OpenGL render functions to capture vertex buffer data, transmitting it via an IO pipe to the Python-based _dAIanna.py_ application.

### Note About Linux Support:
- Because the project has pivoted to using a DLL which is inherently a Windows library type, Linux is currently not officially supported. _dAIanna.py_ itself doesn't require the DLL to run, but a functional substitute will need to be created that is compatible with Linux in the future in order for it to receive any information about the state of the game

## dAIanna.py


In _dAIanna.py_, the received vertex buffer data undergoes 3D environment reconstruction (to do)

Currently, _dAIanna.py_ reconstructs the scene and creates a preview within MatPlotLib. It utilizes the old environment mapping system where it attempts to recreate the environment from a depth map interpreted from a screenshot of the game. After _dAIannaHook.dll_ is functioning this depth map method will be removed and replaced with a procedure for reconstructing the environment from the vertex buffer intercepted directly from the game, rather than estimating the depth of every pixel in a given screenshot.

By separating rendering interception and AI processing, _dAIanna_ reduces 'investigative' overhead to ensure the RL Agent's environment is updated in real time, allowing it to maintain perception even when driving at high speeds.
