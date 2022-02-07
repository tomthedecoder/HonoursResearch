from Environment import Environment
import os
import numpy as np
import wavio
from time import sleep
import simpleaudio as sa
import tkinter as tk


class SignalGenerator:
    """ Lets the user act as a artificial selector, selecting the best sounds from multiple environments, which I
        suppose are meant to be notes"""

    def __init__(self, num_environments):
        self.pop_size = 20
        self.environments_container = []
        for _ in range(num_environments):
            self.environments_container.append((Environment(target_signal=None, pop_size=self.pop_size)))

        self.signal_duration = 2 * np.pi
        self.num_asked = 4

        # file locations of individuals which play the sounds
        self.tl_location = ""
        self.tr_location = ""
        self.bl_location = ""
        self.br_location = ""

        # index pointing to environment currently being hosted by client
        self.environment_index = 0

        # points to individuals within the frames, top left, top right, ...
        self.individuals_in_use = []

        # initialise the window
        self.window_width = 100
        self.window_height = 100
        self.window = tk.Tk()
        self.window.geometry("{}x{}".format(self.window_width, self.window_height))

        # frames
        self.tl_frame = tk.Frame(self.window, highlightbackground="black", highlightthickness=1, width=self.window_width/2, height=self.window_height/2)
        self.tr_frame = tk.Frame(self.window, highlightbackground="black", highlightthickness=1, width=self.window_width/2, height=self.window_height/2)
        self.bl_frame = tk.Frame(self.window, highlightbackground="black", highlightthickness=1, width=self.window_width/2, height=self.window_height/2)
        self.br_frame = tk.Frame(self.window, highlightbackground="black", highlightthickness=1, width=self.window_width/2, height=self.window_height/2)

        # buttons
        self.tl_play = tk.Button(self.tl_frame, text="play", command=self.play_btn_1)
        self.tl_choose = tk.Button(self.tl_frame, text="choose", command=self.choose)
        self.tl_play.pack(side=tk.BOTTOM, anchor=tk.CENTER)
        self.tl_choose.pack(side=tk.BOTTOM, anchor=tk.CENTER)

        self.tr_play = tk.Button(self.tr_frame, text="play", command=self.play_sound)
        self.tr_choose = tk.Button(self.tr_frame, text="choose", command=self.choose)
        self.tr_play.pack(side=tk.BOTTOM, anchor=tk.CENTER)
        self.tr_choose.pack(side=tk.BOTTOM, anchor=tk.CENTER)

        self.bl_play = tk.Button(self.bl_frame, text="play", command=self.play_sound)
        self.bl_choose = tk.Button(self.bl_frame, text="choose", command=self.choose)
        self.bl_play.pack(side=tk.BOTTOM, anchor=tk.CENTER)
        self.bl_choose.pack(side=tk.BOTTOM, anchor=tk.CENTER)

        self.br_play = tk.Button(self.br_frame, text="play", command=self.play_sound)
        self.br_choose = tk.Button(self.br_frame, text="choose", command=self.choose)
        self.br_play.pack(side=tk.BOTTOM, anchor=tk.CENTER)
        self.bl_choose.pack(side=tk.BOTTOM, anchor=tk.CENTER)

        self.tl_frame.place(x=0, y=0)
        self.tr_frame.place(x=self.window_width, y=self.window_height)
        self.bl_frame.place(x=self.window_width, y=0)
        self.br_frame.place(x=0, y=self.window_height)

        self.handle_interaction()

    def update_sound_files(self, num_asked):
        """ Writes sound files of each environment to each own directory.
            num_asked is the number of sounds the method asks from each sound category"""

        sample_rate = 44100
        to_ask = []

        for idx, environment in enumerate(self.environments_container):
            if not os.path.exists("Environment {}".format(idx)):
                os.makedirs("Environment {}".format(idx))
            to_ask.append([])
            for _ in range(num_asked):
                index = np.random.randint(0, environment.pop_size)
                to_ask[-1].append(index)
                individual = environment.individuals[index]
                individual.evaluate(final_t=self.signal_duration)
                output_y = individual.ctrnn.history[-1]
                try:
                    wavio.write("Individual {}".format(index), output_y, sample_rate, sampwidth=1)
                except Exception():
                    print("something went wrong")

        return to_ask

    def play_btn1(self):
        targets_wave = sa.WaveObject.from_wave_file(".wav")

    def play_sound(self):
        """ Plays a sound"""

        targets_wave = sa.WaveObject.from_wave_file(file_location + ".wav")
        play_target = targets_wave.play()
        play_target.wait_done()

    def choose(self, individual_index):
        """ Result of user choosing a sound, triggering reproduction with remaining three"""

        # I want to over write memory location where original individual is stored
        strong_individual = self.individuals_in_use[individual_index]
        environment = self.environments_container[self.environment_index]
        for idx in self.individuals_in_use:
            if idx == individual_index:
                continue

            weak_individual = environment.individuals[idx]
            environment.individuals[individual_index] = strong_individual.microbial_cross_over(weak_individual)

        self.populate_frames()

    def populate_frames(self):
        """ Populates frame objects"""
        self.environment_index = np.random.randint(0, len(self.environments_container))
        self.individuals_in_use = []
        locations = [self.tl_location, self.tr_location, self.bl_location, self.br_location]
        for i in range(4):
            individual_index = np.random.randint(0, self.pop_size)
            locations[i] = "Environment {}\Individual {}".format(self.environment_index, individual_index)
            self.individuals_in_use.append(individual_index)

    def handle_interaction(self):
        """ Keeps the window running and is responsible for handling all foreground functions of the window"""

        self.window.mainloop()


signal = SignalGenerator(1)
