import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from proj_sdsc.algorithm.algo import Algorithm
from proj_sdsc.model.discriminator import Model
from proj_sdsc import config
from typing import List, Union, Callable
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
from collections import namedtuple
torch.cuda.set_device(0)
import matplotlib.pyplot as plt

class CNNModel(nn.Module):
	def __init__(self, device):
		super(CNNModel, self).__init__()
		model = Model().to(device)
		model.load_state_dict(torch.load("/data/conan/model/model_retrained_100.pth"))
		self.conv_block1 = model.conv_block1
		self.conv_block2 = model.conv_block2
		self.conv_block3 = model.conv_block3
		self.conv_block4 = model.conv_block4
		self.outputs = namedtuple("Outputs", ['conv1', 'conv2', 'conv3', 'conv4'])
	def forward(self, x):
		conv_block1 = self.conv_block1(x)
		conv_block2 = self.conv_block2(conv_block1)
		conv_block3 = self.conv_block3(conv_block2)
		conv_block4 = self.conv_block4(conv_block3)
		out = self.outputs(conv_block1, conv_block2, conv_block3, conv_block4)
		return out

class GramMatrix(nn.Module):
	def forward(self, input):
		a, b, c = input.size()  # a=batch size(=1)
	# b=number of feature maps
	# (c,d)=dimensions of a f. map (N=c*d)
		features = input.view(a * b, c)  # resise F_XL into \hat F_XL
		G = torch.mm(features, features.t())  # compute the gram product
	# we 'normalize' the values of the gram matrix
	# by dividing by the number of element in each feature maps.
		return G.div(a * b * c)

def total_variation_loss(x):
    img_nrows, img_ncols = x.shape[1], x.shape[2]
    a = torch.square(
        x[:, : img_nrows - 1, : img_ncols - 1] - x[:, 1:, : img_ncols - 1]
    )
    b = torch.square(
        x[:, : img_nrows - 1, : img_ncols - 1] - x[:, : img_nrows - 1, 1:]
    )
    return torch.sum(torch.pow(a + b, 1.25))


def get_loss(model, gram, opti_img, content_features, gram_style, alpha, beta, tv_w):

	cnt_features = [opti_img, *model(opti_img.unsqueeze(0))]

	loss_c = 0.0
	# for cnt, target in zip(cnt_features, content_features):
	# 	loss_c += nn.MSELoss(reduction="sum")(cnt, target)
	# loss_c = nn.MSELoss(reduction="mean")(opti_img, content_features)
	c_feature_choice = 0
	loss_c = nn.MSELoss(reduction="mean")(cnt_features[c_feature_choice], content_features[c_feature_choice])
	loss_c *= alpha/4

	cnt_features = cnt_features[1:]

	loss_s = 0.0
	for cnt, target in zip(cnt_features, gram_style):
		gram_cnt = gram(cnt.squeeze())
		loss_s += nn.MSELoss(reduction="sum")(gram_cnt, target)
	loss_s *= beta/4 # weights all the layers equally
	loss_s /= 4**2 * content_features[c_feature_choice].shape[1]**2
	# normalize by the number of layers squared time the number of channels squared
	loss_tv = tv_w*total_variation_loss(opti_img)

	loss = loss_c + loss_s + loss_tv
	
	return loss, loss_c, loss_s, loss_tv

def save_run(optimizing_img, i):
	np.save("runs/gatys_{}.npy".format(i), optimizing_img.squeeze().detach().cpu().numpy())


class Gatys(Algorithm):
	def __init__(self):
		super(Gatys, self).__init__()


	def algorithm(self, content:torch.Tensor, style:Union[torch.Tensor,int, None], iterations, opti, log_time, alpha=1e-3, beta=1, tv = 0, start_from_content=False) -> torch.Tensor:
		content = content.to(config.device)
		style = style.to(config.device)
		
		init_image = torch.from_numpy(np.random.normal(loc=0, scale=90, size = content.size())).float().to(config.device) if not start_from_content else content.clone()
		optimizing_img = Variable(init_image, requires_grad=True)

		model = CNNModel(config.device)
		total_losses, total_losses_c, total_losses_s, total_losses_tv = [], [], [], []

		content_features = [content, *model(content.unsqueeze(0))]
		style_features = model(style.unsqueeze(0))

		gram = GramMatrix()
		gram.to(config.device)
		gram_style = [gram(cnt.squeeze()).to(config.device) for cnt in style_features]
		img_300, img_3000 = None, None

		if opti == "adam":
			adam = optim.Adam([optimizing_img], lr=1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

			def tuning_step(optimizing_img):
				adam.zero_grad()
				total_loss, loss_c, loss_s, loss_tv = get_loss(model, gram, optimizing_img, content_features, gram_style, alpha=alpha, beta=beta, tv_w=tv)
				total_loss.backward(retain_graph=True)
				adam.step()
				return total_loss, loss_c, loss_s, loss_tv
		
			for i in range(iterations):
				total_loss, loss_c, loss_s, loss_tv = tuning_step(optimizing_img)
				total_losses.append(total_loss.item())
				total_losses_c.append(loss_c.item())
				total_losses_s.append(loss_s.item())
				total_losses_tv.append(loss_tv.item())

				if i % log_time == 0:
					with torch.no_grad():
						print("Iteration: {}, Total Loss: {}, Content Loss: {}, Style Loss: {}, TV Loss: {}".format(i, total_loss.item(), loss_c.item(), loss_s.item(), loss_tv.item()))
				
				if i == 300:
					img_300 = optimizing_img.clone()

				if i == 3000:
					img_3000 = optimizing_img.clone()

			# save_run(optimizing_img, i)

		elif opti == "lbfgs":
			lbfgs = optim.LBFGS((optimizing_img,), max_iter=iterations, line_search_fn='strong_wolfe')
			count = 0
			def closure():
				nonlocal count, total_losses
				total_loss, loss_c, loss_s, _ = get_loss(model, gram, optimizing_img, content_features, gram_style, alpha=alpha, beta=beta, tv_w=tv)
				total_losses.append(total_loss.item())
				lbfgs.zero_grad() 
				total_loss.backward(retain_graph=True)
				if count % log_time == 0:
						with torch.no_grad():
							print("Iteration: {}, Total Loss: {}, Content Loss: {}, Style Loss: {}".format(count, total_loss.item(), loss_c.item(), loss_s.item()))
							# save_run(optimizing_img, count)
				count += 1
				return total_loss
			lbfgs.step(closure,)
		else:
			raise ValueError("Optimization algorithm not recognized")
		
		return optimizing_img, total_losses, total_losses_c, total_losses_s, total_losses_tv, img_300, img_3000

if __name__ == "__main__":
	gatys = Gatys()
	# transform = lambda tensor: tensor.float().T.unsqueeze(0).T
	# content = transform(torch.from_numpy(np.load("/data/conan/spectrogram_dataset/12Le Père Fouettard - Piano/2.npy")))
	# style = transform(torch.from_numpy(np.load("/data/conan/spectrogram_dataset/Fantasia - Guitar/32.npy")))
	content = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/12Le Père Fouettard - Piano/2.npy")).unsqueeze(0)
	style = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/Fantasia - Sax/32.npy")).unsqueeze(0)
	# CONTENT_FILENAME = "/home/conan/neural-style-audio-tf/inputs/imperial.mp3"
	# STYLE_FILENAME = "/home/conan/neural-style-audio-tf/inputs/usa.mp3"
	
	# N_FFT = 2048
	# def read_audio_spectum(filename):
	# 	x, fs = librosa.load(filename)
	# 	S = librosa.stft(x, n_fft=N_FFT)
	# 	p = np.angle(S)
		
	# 	S = np.log1p(np.abs(S[:,:430]))  
	# 	return S, fs
	# a_content, fs = read_audio_spectum(CONTENT_FILENAME)
	# a_style, fs = read_audio_spectum(STYLE_FILENAME)

	# N_SAMPLES = a_content.shape[1]
	# N_CHANNELS = a_content.shape[0]
	# a_style = a_style[:N_CHANNELS, :N_SAMPLES]
	# content = torch.from_numpy(a_content).unsqueeze(0)
	# style = torch.from_numpy(a_style).unsqueeze(0)

	gatys.transfer_style(content, style, iterations=2000, opti="adam", log_time=100, alpha=1, beta=1000)
	out_img = gatys.output[0]
	out_img = out_img.squeeze().detach().cpu().numpy()
	np.save("out.npy", out_img)
	plt.imshow(out_img)
	plt.savefig("out.png")
