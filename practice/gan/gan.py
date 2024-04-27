# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
bs = 128 
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='./mnist_dataset/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_dataset/', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
train_iter = iter(train_loader)
X, y = next(train_iter)
# print(X.shape)
# print(y.shape)

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# build network
z_dim = 64
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
# print(G)
D = Discriminator(mnist_dim).to(device)
# print(D)

# loss
criterion = nn.BCELoss() 

# optimizer
lr = 0.001
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()
    cur_batch_size = x.size(0)

    # Discriminator real loss
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(cur_batch_size, 1)

    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Discriminator fake loss
    z = torch.randn(cur_batch_size, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(cur_batch_size, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # Discriminator optimization
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()
    cur_batch_size = x.size(0)

    z = torch.randn(cur_batch_size, z_dim).to(device)
    y = torch.ones(cur_batch_size, 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

     # Generator optimization
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()



start_time = time.time()
n_epoch = 3
D_epoch_loss, G_epoch_loss = [], []

for epoch in range(1, n_epoch+1):
    D_losses, G_losses = [], []

    for batch_idx, (x, y) in enumerate(train_loader):

        # print(f"Train_loader: batch_no:{batch_idx}, Input shape:{x.shape}, Label shape:{y.shape}")
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    d_loss_mean = torch.mean(torch.FloatTensor(D_losses))
    g_loss_mean = torch.mean(torch.FloatTensor(G_losses))

    D_epoch_loss.append(d_loss_mean)
    G_epoch_loss.append(g_loss_mean)

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch), n_epoch, d_loss_mean, g_loss_mean))

print(D_epoch_loss)
print(G_epoch_loss)

print(f"Train Time:{time.time()-start_time}")

# with torch.no_grad():
#     test_z = Variable(torch.randn(bs, z_dim).to(device))
#     generated = G(test_z)
#     save_image(generated.view(generated.size(0), 1, 28, 28), './result/gan_sample_' + '.png')

