import numpy as np
from pydub import AudioSegment
from OutputHandler import *
from pydub.playback import play
from scipy.io import wavfile


class AudioHandler:
    def __init__(self, start_t=0.0, final_t=1.4):
        self.start_t = start_t
        self.final_t = final_t

        self.end = -1
        self.non_transient_start = 0
        self.time = []
        self.data = []
        self.sample_rate = -1
        self.note_name = ''

        self.data_scaler = OutputHandler('max/min')
        self.time_scaler = OutputHandler('scale&origin')

    def read_from_file(self, filename: str):
        self.note_name = filename[:filename.find(".")].strip()
        self.sample_rate, self.data = wavfile.read(filename)

        self.end = min(int(self.final_t * self.sample_rate), len(self.data) - 1)
        self.non_transient_start = min(int(self.start_t * self.sample_rate), len(self.data) - 1)

        length = self.data.shape[0]/self.sample_rate
        self.time = np.linspace(0, length, self.data.shape[0])

        return self.time, self.data

    def convert(self, audio_file: str, new_format: str):
        format = audio_file[audio_file.find(".")+1:].strip()
        name = audio_file[:audio_file.find(".")].strip()
        caudio = AudioSegment.from_file(audio_file, format=format)
        caudio.export(f"{name}.{new_format}", format=new_format)

    def play(self, filename: str):
        format = filename[filename.find(".")+1:].strip()
        sound = AudioSegment.from_file(filename, format=format)
        play(sound)

    def non_transient_audio(self):
        return self.time_scaler.call(self.time[self.non_transient_start:self.end]), self.data_scaler.call(self.data[self.non_transient_start:self.end])

    def plot_audio(self):
        import matplotlib.pyplot as plt

        time, non_transient_audio = self.non_transient_audio()
        plt.plot(time, non_transient_audio)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(f"Non-Transient Audio Of {self.note_name}")
        plt.grid()

        plt.figure()

        plt.plot(self.time, self.data)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(self.note_name)
        plt.grid()

        plt.show()


def play_audio():
    from os.path import exists

    filename = "Piano A4"
    audio_handler = AudioHandler(0.8, 0.826)
    if not exists(f"{filename}.wav") and exists(f"{filename}.m4a"):
        audio_handler.convert(f"{filename}.m4a", "wav")

    audio_handler.read_from_file(f"{filename}.wav")
    audio_handler.plot_audio()
    #audio_handler.play(f"{filename}.wav")

#play_audio()