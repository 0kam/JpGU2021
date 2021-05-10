import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from scripts.utils import read_sses, DrawDS, load_npy, LabelledDS, cf_labelled, cf_unlabelled, draw_legend, draw_teacher
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from glob import glob
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image

class LSTMClassifier(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(LSTMClassifier, self).__init__()
        self.h_dim = int((x_dim + y_dim) / 2)
        self.lstm = nn.LSTM(x_dim, self.h_dim, batch_first=True)
        self.bn1 = nn.BatchNorm1d(self.h_dim)
        self.do1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.h_dim, self.h_dim)
        self.prelu = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(self.h_dim)
        self.do2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.h_dim, y_dim)

    def forward(self, x):
        _, h = self.lstm(x)
        h = h[0].view(-1, self.h_dim)
        h = self.bn1(h)
        h = self.do1(h)
        h = self.prelu(self.fc1(h))
        h = self.bn2(h)
        h = self.do2(h)
        return F.softmax(self.fc2(h), dim=1)

class SimpleLSTM():
    def __init__(self, data_dir, labels_dir, batch_size,device="cuda", num_workers=20, label="all", shrink=0.0):
        self.classes = [Path(n).name for n in glob(data_dir + "/labelled/*")]
        self.device = device
        _, labels = read_sses(labels_dir, (9999,9999), label=label)
        self.label = label
        self.shrink = shrink
        self.labels = labels
        self.labels_dir = labels_dir
        ## labelled data loader
        labelled = LabelledDS(data_dir + "/labelled/")
        train_indices, val_indices = train_test_split(list(range(len(labelled.dataset.targets))), test_size=0.2, stratify=labelled.dataset.targets)
        train_dataset = torch.utils.data.Subset(labelled, train_indices)
        val_dataset = torch.utils.data.Subset(labelled, val_indices)
        x, y = train_dataset[0]
        self.x_dim = x.shape[2]
        self.y_dim = len(self.classes)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=cf_labelled)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cf_labelled)
        self.class_to_idx = self.train_loader.dataset.dataset.dataset.class_to_idx

        self.model = LSTMClassifier(self.x_dim, self.y_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_cls = nn.CrossEntropyLoss()
        self.best_test_loss = 9999
    
    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = torch.eye(self.y_dim)[y].to(self.device)
            self.model.zero_grad()
            y2 = self.model(x)
            loss = self.loss_cls(y2, y.argmax(1))
            loss.backward()
            self.optimizer.step()

            train_loss += loss
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss
    
    def _val(self, epoch):
        self.model.eval()
        test_loss = 0
        total = [0 for _ in range(len(self.classes))]
        tp = [0 for _ in range(len(self.classes))]
        fp = [0 for _ in range(len(self.classes))]
        for x, _y in self.val_loader:
            x = x.to(self.device)
            y = torch.eye(self.y_dim)[_y].to(self.device)
            with torch.no_grad():
                y2 = self.model(x)
            loss = self.loss_cls(y2, y.argmax(1))
            test_loss += loss
            y2 = y2.argmax(1)
            for c in range(len(self.classes)):
                pred_yc = y2[_y==c]
                _yc = _y[y2==c]
                total[c] += len(_y[_y==c])
                tp[c] += len(pred_yc[pred_yc==c])
                fp[c] += len(_yc[_yc!=c])
        
        test_loss = test_loss * self.val_loader.batch_size / len(self.val_loader.dataset)
        test_recall = [100 * c / t for c,t in zip(tp, total)]
        test_precision = []
        for _tp,_fp in zip(tp, fp):
            if _tp + _fp == 0:
                test_precision.append(0)
            else:
                test_precision.append(100 * _tp / (_tp + _fp))
        c = self.train_loader.dataset.dataset.dataset.class_to_idx
        recall = {}
        prec = {}
        for _, row in self.labels.iterrows():
            index = row[0]
            name = row[1]
            recall[name] = test_recall[c[str(index)]]
            prec[name] = test_precision[c[str(index)]]
        
        print("Test Loss:", str(test_loss), "Test Recall:", str(recall), "Test Precision:", str(prec))
        return test_loss, recall, prec
    
    def train(self, epochs, log_dir):
        writer = SummaryWriter("./runs/" + log_dir)
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            val_loss, recall, precision = self._val(epoch)
            if val_loss < self.best_test_loss:
                self.best_model = LSTMClassifier(self.x_dim, self.y_dim).to(self.device)
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_recall = recall
                self.best_prec = precision
                self.best_test_loss = val_loss
            writer.add_scalar("test_loss", val_loss, epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            for label in recall:
                writer.add_scalar("test_recall_" + label, recall[label], epoch)
                writer.add_scalar("test_precision_" + label, precision[label], epoch)
    
    def draw(self, image_dir, out_path, kernel_size, batch_size):
        with Image.open(glob(image_dir+"/*")[0]) as img:
            w, h = img.size
        dataset = DrawDS(image_dir, kernel_size)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        confs = []
        self.best_model.eval()
        with torch.no_grad():
            for x in tqdm(loader):
                x = x.to(self.device)
                y = self.best_model(x)
                pred_y = y.argmax(1).detach().cpu()
                conf = y.max(1)[0].detach().cpu()
                pred_ys.append(pred_y)
                confs.append(conf)
        
        seg_image = torch.cat(pred_ys).reshape([h,w]).numpy()
        confs = torch.cat(confs).reshape([h,w]).numpy()
        cmap = plt.get_cmap("tab20", len(self.classes))
        plt.imsave(out_path, seg_image, cmap = cmap)
        return seg_image, confs
    
    def draw_teacher(self, out_path, image_size):
        draw_teacher(out_path, self.labels_dir, self.class_to_idx, image_size, self.label, self.shrink)
    
    def draw_legend(self, out_path):
        draw_legend(out_path, self.labels_dir, self.class_to_idx, self.label)