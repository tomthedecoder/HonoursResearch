from AudioHandler import *
from scipy.interpolate import interp1d
import numpy as np


class TargetSignal:
    def __init__(self, start_t: float, final_t: float,  type='signal', signal=None, filename='Piano A4.wav'):
        assert type != 'signal' or callable(signal)

        self.audio_handler = AudioHandler(start_t, final_t)
        self.filename = filename
        self.type = type

        # store audio contents into self.audio_handler
        self.audio_handler.read_from_file(self.filename)
        self.times, self.y_values = self.audio_handler.non_transient_audio()

        self.step_size = self.times[1] - self.times[0]

        self.signal = signal if type == 'signal' else interp1d(self.times, self.y_values, kind='linear')

    def __call__(self, *args):
        assert len(args) == 1

        return self.signal(args[0])

    def __str__(self):
        return f"{self.signal}"


