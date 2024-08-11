import os, sys, torch

#This is the brains of the dAIanna AI

action_table = [
    {"gas": 0, "brake": 0, "left": 0, "right": 0},  # No Action
    {"gas": 1, "brake": 0, "left": 0, "right": 0},  # Gas
    {"gas": 0, "brake": 1, "left": 0, "right": 0},  # Brake
    {"gas": 1, "brake": 0, "left": 1, "right": 0},  # Gas + Left
    {"gas": 1, "brake": 0, "left": 0, "right": 1},  # Gas + Right
    {"gas": 0, "brake": 1, "left": 1, "right": 0},  # Brake + Left
    {"gas": 0, "brake": 1, "left": 0, "right": 1}   # Brake + Right
]

action_id = model_output
action = action_table[action_id]

print("Neurons test")