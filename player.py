"""Plays one sound file. Used by main_v5.py.
Usage: python player.py <sound_file> [--loop]
Uses the same pygame.mixer.Sound method as the original play.py.
"""
import os
import sys
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

if len(sys.argv) < 2:
    print("Usage: python player.py <sound_file> [--loop]")
    sys.exit(1)

path = sys.argv[1]
loop = "--loop" in sys.argv

pygame.mixer.init()
sound = pygame.mixer.Sound(path)
sound.play(loops=-1 if loop else 0)

while pygame.mixer.get_busy():
    time.sleep(0.1)