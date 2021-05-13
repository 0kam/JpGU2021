from scripts.utils import set_patches
from scripts.cnn_lstm import CNNLSTM
set_patches("data/train2/2010/labels", "data/train2/2010/images_2010", "data/train2/patches9x9/", (9,9), batch_size=50, unlabelled=False, shrink=0)
set_patches("data/train2/2010/labels", "data/train2/2020/images_2020", "data/train2/patches9x9/", (9,9), batch_size=50, unlabelled=False, shrink=5)

lstm = CNNLSTM("data/train2/patches9x9", "data/train2/2010/labels", (9,9), 200, device="cuda", shrink = 5)
lstm.draw_legend("results/legend.png")
lstm.draw_teacher("results/teacher_2010_shrink5.png", (5616,3744))

lstm.train(1500, "cnn_lstm")

res2010, _ = lstm.draw("data/train2/2010/images_2010", "results/simple_lstm_9x9_2010.png", (9,9), 50000)
res2020, _ = lstm.draw("data/train2/2020/images_2020", "results/simple_lstm_9x9_2020.png", (9,9), 50000)

lstm.draw_legend("legend.png")
lstm.draw_teacher("teacher_2015_full_shrink01.png", (5616,3744))

import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
import cv2
np.save("results/res2010_9x9.npy", res2010)
np.save("results/res2020_9x9.npy", res2020)

lstm.labels
lstm.class_to_idx

img2010 = cv2.imread("data/pictures/mrd_085_eos_vis_20101023_1200.png")[:,:,0]
img2020 = cv2.imread("data/pictures/mrd_085_eos_vis_20201021_1205.png")[:,:,0]
mask = np.logical_and(img2010!=0, img2020!=0).astype(int)

diff = np.zeros(res2010.shape)
diff[np.logical_and(res2010!=4, res2020==4)] = 1 # ハイマツ以外 -> ハイマツ
diff[np.logical_and(res2010==4, res2020!=4)] = 2 # ハイマツ -> ハイマツ以外
cmap = plt.get_cmap("tab20", 3)
diff = diff * mask
plt.imsave("results/diff_haimatsu_9x9.png", diff, cmap = cmap)

diff_labels = [
    "出現",
    "消滅"
]
pl.cla()
for i, l in zip(range(2), diff_labels):
    color = cmap(i+1)
    pl.plot(0, 0, "-", c = color, label = l, linewidth = 10)
pl.legend(loc = "center", prop = {"family": "MigMix 1P"})
pl.savefig("results/diff_legend.png")

diff = np.zeros(res2010.shape)
diff[np.logical_and(res2010!=0, res2020==0)] = 1 # ササ以外 -> ササ
diff[np.logical_and(res2010==0, res2020!=0)] = 2 # ササ -> ササ以外 
diff = diff * mask
plt.imsave("results/diff_sasa_9x9.png", diff, cmap = cmap)