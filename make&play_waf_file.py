from scipy.io.wavfile import write
import simpleaudio as sa
import numpy as np
from playsound import playsound
from time import sleep
from read_from_file import *
import wavio


def play_sound():
    """ Play each sound"""
    print("The target sound")
    targets_wave = sa.WaveObject.from_wave_file("target_sound.wav")
    play_target = targets_wave.play()
    play_target.wait_done()
    sleep(1)
    print("Output sound")
    outputs_wave = sa.WaveObject.from_wave_file("output_sound.wav")
    play_output = outputs_wave.play()
    play_output.wait_done()




target_y = read_from_file("targets_file")
output_y = read_from_file("output_file")
time = read_from_file("time_file")

sample_rate = 44100

wavio.write("target_sound.wav", target_y, sample_rate, sampwidth=1)
wavio.write("output_sound.wav", output_y, sample_rate, sampwidth=1)

play = True
if play:
    play_sound()