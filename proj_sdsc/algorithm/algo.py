from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor, from_numpy as to
from typing import Union, Callable
from scipy.io import wavfile
import librosa

class Algorithm(ABC):
    num_steps = 1000
    # @classmethod
    def transfer_style(self, content:Tensor, style:Union[Tensor,int],  iterations:int=1000, progress_bar:Union[Callable, None] =None, opti =None, log_time=500, alpha=1e-3, beta=1, tv = 0) -> Tensor:
        assert content is not None, 'content must be loaded before hand or file is empty'
        assert style is not None, 'style must be loaded before hand or file is empty'
        

        self.output = self.algorithm(content, style=style,iterations=iterations, opti=opti, log_time=log_time, alpha=alpha, beta=beta, tv=tv)

    @abstractmethod
    def algorithm(self, content:Tensor, style:Union[Tensor,int], iterations, opti, log_time) -> Tensor:
        pass

    def save_output(self, save_path:str):
        assert self.output
        assert save_path.endswith('.npy'), 'save_path must be a .npy file'
        np.save(save_path, self.output)
        self.output_path = save_path

    def load_from_audio(self, audio_path:str) -> None:
        assert audio_path.endswith(('mp3', 'wav')), 'audio_type must be mp3 or wav'
        audio_type = audio_path.split('.')[-1]

        if audio_type == 'mp3':
            y, _ = librosa.load(audio_path)
            D = librosa.stft(y, n_fft=2048)
            S, _ = librosa.magphase(D)
            content = librosa.power_to_db(S, ref=np.max)
            
        elif audio_type == 'wav':
            _, content = wavfile.read(audio_path)
        else:
            raise ValueError('audio_type must be mp3 or wav')

        assert content.any(), 'content must be loaded or file is empty' 
  
        self.content = to(content).unsqueeze(0)

    def load_from_array(self, array_path:str):
        assert array_path.endswith('.npy'), 'array_path must be a .npy file'
        self.content = to(np.load(array_path))

    def get_output(self):
        return np.load(self.output_path)


    def get_output_path(self):
        return self.output_path