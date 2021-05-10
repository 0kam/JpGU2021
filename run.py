from scripts.utils import set_patches
from scripts.simple_lstm import SimpleLSTM
#set_patches("data/train/2015/labels", "data/train/2015/images", "data/train/2015/patches/", (3,3), batch_size=50, unlabelled=False, shrink=0.0)

lstm = SimpleLSTM("data/train/2015/patches", "data/train/2015/labels", 200, device="cpu", shrink = 10)
lstm.train(500, "simple_lstm")

lstm.draw("data/2015/source/", "simple_lstm_3x3_2015.png", (3,3), 50000)
lstm.draw("data/2010/source", "simple_lstm_3x3_2010.png", (3,3), 5000)
lstm.draw("data/2020/source", "simple_lstm_3x3_2020.png", (3,3), 5000)

lstm.draw_legend("legend.png")
lstm.draw_teacher("teacher_2015_full_shrink01.png", (5616,3744))

from scripts.gmvae import GMVAE
gmvae = GMVAE("data/2015/patches", "data/2015/labels", 2000)
gmvae.train(50, "gmavae")