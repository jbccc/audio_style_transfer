import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch

from proj_sdsc.model.dataset import LitDataModule
from proj_sdsc import config
from proj_sdsc.model.discriminator import Model, newVGG, UlyNet
import numpy as np
from tqdm import tqdm

def test(model:torch.nn.Module, test_data_module,device):
	print('start eval...')
	## test the model accuracy on the test set
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data, target in test_data_module:

			if data.shape[0]<32:
				continue
			
			data:torch.Tensor = data.reshape(32, 1, 1025, 431)
			target:torch.Tensor = target.argmax(dim=1)
			data, target = data.to(device), target.to(device)

			outputs:torch.Tensor = model(data)
			predicted = outputs.argmax(dim=1)
			total += target.size(0)
			correct += (predicted == target).sum().item()

	print('Accuracy of the network on the test images: %d %%' % (
		100 * correct / total))
	return correct / total


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# model = Model()
	# model.load_state_dict(torch.load("/data/conan/model/model_60_ep.pth"))
	N_SAMPLES = 430
	N_CHANNELS = 1025
	N_FILTERS = 4096
	std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
	kernel_data = np.random.randn(1, 11, N_CHANNELS, N_FILTERS) * std
	model = UlyNet(kernel_data)
	model.load_state_dict(torch.load("/data/conan/model/modelVGG.pth"))
	model.to(device)
	lit_module = LitDataModule(config.dataset["spectrogram"], 32)
	lit_module.setup()
	test_data_module = lit_module.train_dataloader()

	test(model=model, test_data_module=test_data_module, device=device)