from scripts.distributions_gru import Generator, Inference, Classifier, Prior, Prior_Q
from scripts.model import Tissue
from pixyz.losses import ELBO, KullbackLeibler, Expectation, LogProb
from pixyz.models import Model
from pixyz.distributions import Normal
import torch
import torch_optimizer as optim

class GMVAE(Tissue):
    def __init__(self, patch_dir, label_dir, z_dim=4, batch_size=20, drop_out_rate=0, lr=1e-3, device="cuda", num_workers=1):
        super().__init__(patch_dir, label_dir, z_dim, batch_size, drop_out_rate, lr, device, num_workers)
        
        self.p = Generator(self.x_dim, z_dim, self.seq_length, device).to(device)
        self.q = Inference(self.x_dim, self.y_dim, z_dim, drop_out_rate=drop_out_rate).to(device)
        self.f = Classifier(self.x_dim, y_dim=len(self.classes), drop_out_rate=drop_out_rate).to(device)
        self.prior = Prior(z_dim, y_dim=len(self.classes)).to(device)
        # self.prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
        #       var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)
        self.p_joint = self.p * self.prior
        
        # distributions for unsupervised learning
        _q_u = self.q.replace_var(x="x_u", y="y_u").to(device)
        p_u = self.p.replace_var(x="x_u").to(device)
        f_u = self.f.replace_var(x="x_u", y="y_u").to(device)
        #prior_u = self.prior.replace_var(y="y_u").to(device)
        q_u = _q_u * f_u
        p_joint_u = p_u * self.prior
        
        elbo_u = ELBO(p_joint_u, q_u)
        elbo = ELBO(self.p_joint, self.q)
        nll = -self.f.log_prob() #-LogProb(f)
        
        rate = 1 * (len(self.unlabelled_loader) + len(self.train_loader)) / len(self.train_loader)
        
        self.loss_cls = -elbo_u.mean() - elbo.mean() + (rate * nll).mean() 
        self.loss_cls_test = nll.mean()
        self.model = Model(self.loss_cls,test_loss=self.loss_cls_test,
                      distributions=[self.p, self.q, self.f, self.prior], optimizer=optim.RAdam, optimizer_params={"lr":lr})
        print("Model:")
        print(self.model)