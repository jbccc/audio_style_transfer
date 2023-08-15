import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from proj_sdsc.algorithm.gatys import Gatys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

tic = time.time()

content_image = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/12Le Père Fouettard - Piano/2.npy")).reshape(1, 431, 1025)
style_image = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/Fantasia - Sax/32.npy")).reshape(1, 431, 1025)

gatys = Gatys()

# gather losses for the 10'000 first iterations
print("Running Gatys algorithm for 10'000...")
optimizing_img, total_losses, total_losses_c, total_losses_s, _, img_300, img_3000 = gatys.algorithm(
    content_image, style_image, iterations=10000, opti="adam", log_time = 100, alpha=1e-2, beta=1e-1
)
os.chdir("tests/ST_checks")
print("Done. Losses gathered")
# plot losses for 300, 3'000 then 10'000 iterations
plt.plot(total_losses[:300], label='total')
plt.plot(total_losses_c[:300], label='content')
plt.plot(total_losses_s[:300], label='style')
plt.title('Losses for 300 first iterations')
plt.legend()
plt.savefig("ST_losses_300.png")
plt.clf()

plt.imshow(img_300.squeeze(0).squeeze(0).detach().cpu().numpy())
plt.title("Resulting image")
plt.savefig("ST_result_300.png")
plt.clf()
np.save("ST_result_300.npy", img_300.squeeze(0).squeeze(0).detach().cpu().numpy())

plt.plot(total_losses[:3000], label='total')
plt.plot(total_losses_c[:3000], label='content')
plt.plot(total_losses_s[:3000], label='style')
plt.title('Losses for 3000 first iterations')
plt.legend()
plt.savefig("ST_losses_3000.png")
plt.clf()

plt.imshow(img_3000.squeeze(0).squeeze(0).detach().cpu().numpy())
plt.title("Resulting image")
plt.savefig("ST_result_3000.png")
plt.clf()
np.save("ST_result_3000.npy", img_3000.squeeze(0).squeeze(0).detach().cpu().numpy())

plt.plot(total_losses, label='total')
plt.plot(total_losses_c, label='content')
plt.plot(total_losses_s, label='style')
plt.title('Losses for 10000 first iterations')
plt.legend()
plt.savefig("ST_losses_10000.png")
plt.clf()

plt.imshow(optimizing_img.squeeze(0).squeeze(0).detach().cpu().numpy())
plt.title("Resulting image")
plt.savefig("ST_result_10000.png")
plt.clf()
np.save("ST_result_10000.npy", optimizing_img.squeeze(0).squeeze(0).detach().cpu().numpy())

print("Image saved")

print("Running Gatys algorithm for 1'000 with LBFGS...")
_, total_losses_lbfgs, _, _, _, _, _ = gatys.algorithm(
    content_image, style_image, iterations=1000, opti="lbfgs", log_time = 100, 
)

print("Done. Losses gathered")

plt.plot(total_losses_lbfgs, label='total-lbfgs')
plt.plot(total_losses, label='total-adam')
plt.title('Losses for 10000 first iterations')
plt.legend()
plt.savefig("ST_losses_adam_vs_lbfgs.png")
plt.clf()

# gather losses for the 3'000 first iterations with Total Variation
print("Running Gatys algorithm for 3000 with TV...")
optimizing_img, total_losses, total_losses_c, total_losses_s, total_losses_tv, _, _ = gatys.algorithm(
    content_image, style_image, iterations=3000, opti="adam", log_time = 100, tv = 1e-6
)
print("Done. Losses gathered")
plt.plot(total_losses, label='total')
plt.plot(total_losses_c, label='content')
plt.plot(total_losses_s, label='style')
plt.plot(total_losses_tv, label='tv')
plt.title('Losses for 3000 first iterations')
plt.legend()
plt.savefig("ST_losses_3000tv.png")
plt.clf()

plt.imshow(optimizing_img.squeeze(0).squeeze(0).detach().cpu().numpy())
plt.title("Resulting image")
plt.savefig("ST_result_3000tv.png")
plt.clf()
np.save("ST_result_3000tv.npy", optimizing_img.squeeze(0).squeeze(0).detach().cpu().numpy())
print("Images saved")

# starting from content image
print("Running Gatys algorithm for 3000 from content image...")
optimizing_img, total_losses, total_losses_c, total_losses_s, total_losses_tv, _, _ = gatys.algorithm(
    content_image, style_image, iterations=3000, opti="adam", log_time = 100, tv = 0, start_from_content = True
)
print("Done. Losses gathered")
plt.plot(total_losses, label='total')
plt.plot(total_losses_c, label='content')
plt.plot(total_losses_s, label='style')
plt.plot(total_losses_tv, label='tv')
plt.title('Losses for 3000 first iterations')

plt.legend()
plt.savefig("ST_losses_3000content.png")
plt.clf()

plt.imshow(optimizing_img.squeeze(0).squeeze(0).detach().cpu().numpy())
plt.title("Resulting image")
plt.savefig("ST_result_3000content.png")
plt.clf()

print(f"Running finished in { (time.time() - tic)/60 } minutes")