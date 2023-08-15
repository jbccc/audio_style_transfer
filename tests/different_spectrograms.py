import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse
N_FFT = 2048
N_MELS=1024

def spectrogram1(y):
	D = librosa.stft(y, n_fft=N_FFT)
	S, phase = librosa.magphase(D)
	S = librosa.power_to_db(S, ref=np.max)
	return S

def spectrogram2(y):
    S = librosa.stft(y, n_fft=N_FFT)
    S = np.log1p(np.abs(S))  
    return S


def spectrogram3(y, sr):
    mel_S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    mel_S_db = librosa.power_to_db(mel_S, ref=np.max)
    return mel_S_db

def spectrogram4(y,sr):
	mel_S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, n_mels=N_MELS)
	mel_S = np.log1p(np.abs(mel_S))
	return mel_S

def main(args):
    # define 1x3 figure
	fig, axs = plt.subplots(2, 2, figsize=(7, 10))
	# load audio
	y, sr = librosa.load(args.audio)
	# plot spectrogram1
	S = spectrogram1(y)
	axs[0][0].imshow(S)
	axs[0][0].set_title('STFT power_to_db')
	# plot spectrogram2
	S = spectrogram2(y)
	axs[0][1].imshow(S)
	axs[0][1].set_title('STFT log1p')
	# plot spectrogram3
	S = spectrogram3(y, sr)
	axs[1][0].imshow(S)
	axs[1][0].set_title('Mel spectrogram power_to_db')
	# plot spectrogram4
	S = spectrogram4(y, sr)
	axs[1][1].imshow(S)
	axs[1][1].set_title('Mel spectrogram log1p')

	plt.savefig(f"diffspec-{args.type}.png")
	plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, help="path to audio file")
    parser.add_argument('--type', type=str, default="piano")
    
    args = parser.parse_args()
    main(args)