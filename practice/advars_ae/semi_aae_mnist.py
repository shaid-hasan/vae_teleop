import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

mnist_path = '/scratch/qmz9mg/vae/data'
dataset = dsets.MNIST(root=mnist_path,train=True, transform=transforms.ToTensor(),  download=True)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#Encoder
class Q_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim, n_class):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
        self.lin3cat = nn.Linear(N, n_class)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)

        xgauss = self.lin3gauss(x)
        xcat = self.lin3cat(x)

        return xgauss, xcat

# Decoder
class P_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

# Discriminator
class D_net_gauss(nn.Module):  
    def __init__(self,N,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))  

class D_net_cat(nn.Module):
    def __init__(self, N, cat_dim):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(cat_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))



z_dims = 120
n_class = 2
Q = Q_net(784,1000,z_dims, n_class).cuda()
P = P_net(784,1000,z_dims).cuda()
D_gauss = D_net_gauss(500,z_dims).cuda()
D_cat = D_net_cat(100, n_class).cuda() 

print(Q)
print(P)
print(D_gauss)
print(D_cat)

# Set learning rates
gen_lr = 0.001
reg_lr = 0.001
reg_lr_cat = 1e-5
EPS = 1e-15

adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D_gauss = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)
optim_D_cat = torch.optim.Adam(D_cat.parameters(), lr=reg_lr_cat)


epochs = 100
Reconstruction_loss = []
Discriminator_loss = []
Generator_loss = []

# for epoch in range(epochs):
#     total_loss = 0

#     for batch_idx, (X, Y) in enumerate(data_loader):

#         X, Y = to_var(X.view(X.size(0), -1)), to_var(Y)
#         # print("X.shape:",X.shape)
#         # print("Y.shape:",Y.shape)

#         P.zero_grad()
#         Q.zero_grad()
#         D_gauss.zero_grad()

#         # Reconstruction Loss and Optimization of Q and P
#         latent_z = Q(X)   
#         X_hat = P(latent_z)
#         recon_loss = F.binary_cross_entropy(X_hat+EPS,X+EPS) 
#         # recon_loss = reconstruction_loss(X_hat,X)
#         recon_loss.backward()
#         optim_P.step()
#         optim_Q_enc.step()


#         # Adversarial Loss and Optimization of D
#         Q.eval()
#         real_gauss_label = torch.ones((X.shape[0], 1), requires_grad=False).cuda()
#         z_real_gauss = (torch.randn(X.size()[0], z_dims) * 5.).cuda()
#         D_real_gauss = D_gauss(z_real_gauss)
        
#         fake_gauss_label = torch.zeros((X.shape[0], 1), requires_grad=False).cuda()
#         z_fake_gauss = Q(X)
#         D_fake_gauss = D_gauss(z_fake_gauss)

#         real_loss = adversarial_loss(D_real_gauss, real_gauss_label)
#         fake_loss = adversarial_loss(D_fake_gauss, fake_gauss_label)
#         D_loss = 0.5*(real_loss + fake_loss)
#         # D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

#         D_loss.backward()
#         optim_D.step()

#         # Adversarial Loss and Optimization of Q
#         Q.train()
#         z_fake_gauss = Q(X)
#         D_fake_gauss = D_gauss(z_fake_gauss)
#         # G_loss = -torch.mean(torch.log(D_fake_gauss)+EPS)
#         G_loss = adversarial_loss(D_fake_gauss, real_gauss_label)
#         G_loss.backward()
#         optim_Q_gen.step()

#     Reconstruction_loss.append(recon_loss.item())
#     Discriminator_loss.append(D_loss.item())  
#     Generator_loss.append(G_loss.item())  
#     print(f"Epoch {epoch+1}: recon_loss: {recon_loss.item():.8f}, D_loss: {D_loss.item():.8f}, G_loss: {G_loss.item():.8f}")


# def plot_losses_subplots_and_save(Reconstruction_loss, Discriminator_loss, Generator_loss, save_path):
#     epochs = range(len(Reconstruction_loss))

#     fig, axs = plt.subplots(3, 1, figsize=(10, 15))

#     axs[0].plot(epochs, Reconstruction_loss, label='Reconstruction Loss', color='blue')
#     axs[0].set_title('Reconstruction Loss')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_ylabel('Loss')
#     axs[0].legend()
#     axs[0].grid(True)

#     axs[1].plot(epochs, Discriminator_loss, label='Discriminator Loss', color='orange')
#     axs[1].set_title('Discriminator Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_ylabel('Loss')
#     axs[1].legend()
#     axs[1].grid(True)

#     axs[2].plot(epochs, Generator_loss, label='Generator Loss', color='green')
#     axs[2].set_title('Generator Loss')
#     axs[2].set_xlabel('Epoch')
#     axs[2].set_ylabel('Loss')
#     axs[2].legend()
#     axs[2].grid(True)

#     plt.tight_layout()
#     plt.savefig(save_path)  # Save the plot
#     plt.close()  # Close the plot to avoid displaying it
    
#     print(f"Plot saved at: {save_path}")

# save_path = "/scratch/qmz9mg/vae/practice/advars_ae/loss_plot.png"
# plot_losses_subplots_and_save(Reconstruction_loss, Discriminator_loss, Generator_loss, save_path)
