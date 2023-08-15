import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from proj_sdsc.algorithm.algo import Algorithm
from proj_sdsc.model.discriminator import Model
from typing import List, Union, Callable
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
import librosa
import soundfile as sf
from collections import namedtuple

class CNNModel(nn.Module):
	def __init__(self, device):
		super(CNNModel, self).__init__()
		model = Model().to(device)
		model.load_state_dict(torch.load("/data/conan/model/model_60_ep.pth"))

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


def get_loss(model, gram, opti_img, content_features, gram_style, alpha, beta):
	cnt_features = model(opti_img.unsqueeze(0))

	loss_c = 0.0
	for cnt, target in zip(cnt_features, content_features):
		loss_c += nn.MSELoss(reduction="sum")(cnt, target)
	loss_c *= alpha/4

	loss_s = 0.0
	for cnt, target in zip(cnt_features, gram_style):
		gram_cnt = gram(cnt.squeeze())
		loss_s += nn.MSELoss(reduction="sum")(gram_cnt, target)
	loss_s *= beta/4

	loss = loss_c + loss_s
	
	return loss, loss_c, loss_s

def save_run(optimizing_img, i):
	np.save("runs/gatys_{}.npy".format(i), optimizing_img.squeeze().detach().cpu().numpy())



def algorithm(content:torch.Tensor, style:Union[torch.Tensor,int, None], iterations, opti, log_time) -> torch.Tensor:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	content = content.to(device)
	style = style.to(device)

	init_image = torch.from_numpy(np.random.normal(loc=0, scale=90, size = content.size())).float().to(device)
	# init_image = content.clone()
	optimizing_img = Variable(init_image, requires_grad=True)

	model = CNNModel(device)
	model.to(device)
	content_features = model(content.unsqueeze(0))
	style_features = model(style.unsqueeze(0))
	gram = GramMatrix()
	gram.to(device)
	gram_style = [gram(cnt.squeeze()).to(device) for cnt in style_features]

	if opti is "adam":
		adam = optim.Adam([optimizing_img], lr=0.03, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
		def tuning_step(optimizing_img):
			total_loss, loss_c, loss_s = get_loss(model, gram, optimizing_img, content_features, gram_style, alpha=1, beta=1000)
			total_loss.backward(retain_graph=True)
			adam.step()
			adam.zero_grad()
			return total_loss, loss_c, loss_s
		for i in range(iterations):
			total_loss, loss_c, loss_s = tuning_step(optimizing_img)
			if i % log_time == 0:
				with torch.no_grad():
					print("Iteration: {}, Total Loss: {}, Content Loss: {}, Style Loss: {}".format(i, total_loss.item(), loss_c.item(), loss_s.item()))
					save_run(optimizing_img, i)
		
		torch.save(model.state_dict(), "model_test.pth")
	elif opti is "lbfgs":
		lbfgs = optim.LBFGS((optimizing_img,), max_iter=1000, line_search_fn='strong_wolfe')
		def closure():
			total_loss, loss_c, loss_s = get_loss(model, gram, optimizing_img, content_features, gram_style, alpha=1, beta=1000)
			lbfgs.zero_grad()
			total_loss.backward(retain_graph=True)
			return total_loss
		lbfgs.step(closure)
		if i % log_time == 0:
				with torch.no_grad():
					print("Iteration: {}, Total Loss: {}, Content Loss: {}, Style Loss: {}".format(i, total_loss.item(), loss_c.item(), loss_s.item()))
					save_run(optimizing_img, i)
	else:
		raise ValueError("Optimization algorithm not recognized")

if __name__ == "__main__":
	content = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/12Le Père Fouettard - Piano/2.npy")).unsqueeze(0)
	style = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/Fantasia - Guitar/32.npy")).unsqueeze(0)

	algorithm(content, style,20000, opti="lbfgs", log_time=1000)