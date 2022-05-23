#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importations

import os
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import mne





class Muse2:
    
    # Initializing board
    def __init__(self, ses_name, path=r'G:\My Drive\Lab\jupyter_notebooks\Neural Interfaces\recordings'):
    
        self.board_id = 38 #BoardIds.BoardIds.MUSE_2_BOARD.value # or BoardIds.NOTION_2_BOARD.value or BoardIds.NOTION_1_BOARD.value
        #self.serial_number = '18721ec44e87864737f236d0be99f56d'
        self.params = BrainFlowInputParams ()
        self.params.board_id = self.board_id
        #self.params.serial_number = self.serial_number
        BoardShim.enable_dev_board_logger ()
        self.board = BoardShim (self.board_id, self.params)
        self.data = np.array(0)
        self.ses_name = ses_name
        self.path = path
        self.expdur = 0

    # start recording

    def recstart(self):
        self.board.prepare_session()
        self.expdur = time.time()
        self.board.start_stream()
        
        
        os.chdir(self.path)
        os.mkdir(self.ses_name)
        os.chdir(self.path + f'/{self.ses_name}')

        f = open("description.txt","w+")
        f.close()
        self.desc = os.getcwd() + '\\description.txt' # Adding description file

        self.start_time = time.time()

    # stop recording and save
    def recstop(self):
        self.data = self.board.get_board_data()
        self.board.stop_stream()
        self.expdur = time.time() - self.expdur
        self.board.release_session()

        self.eeg_channels = BoardShim.get_eeg_channels(BoardIds.BoardIds.MUSE_2_BOARD (38).value)
        self.eeg_data = self.data[self.eeg_channels, :]
        self.eeg_data = self.eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

        # Creating MNE objects from brainflow data arrays
        self.ch_types = ['eeg'] * len(self.eeg_channels)
        self.ch_names = BoardShim.get_eeg_names(BoardIds.MUSE_2_BOARD.value)
        self.sfreq = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types=self.ch_types)
        self.raw = mne.io.RawArray(self.eeg_data, self.info)

        


    def show_desc(self):
        with open(self.desc, 'r') as f:
            print(f.read())
            f.close()

    def add_timestamp(self, input_str):
        timestamp = time.time() - self.start_time
        timestamp = round(timestamp, 2)
        with open(self.desc, 'a') as f:
            f.write(f'\n[@ {timestamp} s]: {input_str}')
            f.close()



def save(obj):
    
    with open(f'{obj.ses_name}.pickle', 'wb') as f:
        pickle.dump(obj, f)

    f.close()

    fig = plt.figure()
    fig = obj.raw.plot_psd(fmax=50)
    fig.savefig('psd', format='jpg')
    

def load(ses_name):
    print(os.getcwd())
    os.chdir(r'G:\My Drive\Lab\jupyter_notebooks\Neural Interfaces\recordings' + f'/{ses_name}')
    print(os.getcwd())
    with open(f'{ses_name}.pickle', mode='rb') as f:
        obj = pickle.load(f)
    
    return obj

