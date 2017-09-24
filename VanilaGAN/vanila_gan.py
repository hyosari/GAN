import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torchvision.datasets
import torchvision.transforms as transforms
import pdb
""" ====================== HYPER PARAMETER ======================= """
lr = 1e-3
end_epoch = 10000
mb_size = 100
""" ===================== DATA LOADER ========================== """

dataset = torchvision.datasets.MNIST(root='./data',
                                    train=True,
                                    transform = transforms.ToTensor(),
                                    download = True)
data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                        batch_size=mb_size,
                                        shuffle =True)

""" ======================== GENERATOR ============================= """

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.fc1 = nn.Linear(100,128,bias=True)
        self.relu = nn.ReLU(inplace= True)
        self.fc2 = nn.Linear(128,28*28,bias=True)

    def forward(self,x):
        out = self.relu(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        return out

""" ======================== DISCRIMINATOR ========================= """

class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()

        self.fc1 = nn.Linear(28*28,128,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128,1,bias=True)

    def forward(self,x):
        out = self.relu(self.fc1(x))
        out = F.sigmoid(self.fc2(out))

        return out

""" ========================== TRAINING =========================== """
c=0
Z_dim = 100
G = G()
D = D()
G_solver = torch.optim.Adam(G.parameters(),lr =lr)
D_solver = torch.optim.Adam(D.parameters(),lr = lr)


ones_label = torch.autograd.Variable(torch.ones(mb_size,1))
zeros_label = torch.autograd.Variable(torch.zeros(mb_size,1))

for epoch in range(end_epoch):

    for i ,(input,target)  in enumerate(data_loader):

        print('epoch : {} batch: {}/{}'.format(epoch,i,len(data_loader)))
        z = torch.autograd.Variable(torch.randn(mb_size,Z_dim))
        input_var = torch.autograd.Variable(input).view(input.size(0),-1)


        G_sample = G(z)
        D_real = D(input_var)
        D_fake = D(G_sample)

        D_loss_real = F.binary_cross_entropy(D_real,ones_label)
        D_loss_fake = F.binary_cross_entropy(D_fake,zeros_label)

        D_loss = D_loss_real + D_loss_fake

        D_solver.zero_grad()
        D_loss.backward(retain_variables=True)
        D_solver.step()

        G_loss = F.binary_cross_entropy(D_fake,ones_label)

        G_solver.zero_grad()
        G_loss.backward()
        G_solver.step()

        #Print and Plot every now and then
        if epoch % 10 == 0 :
            print('epoch {}/{},D_loss: {},G_loss: {}'.format(epoch,end_epoch,
                                                            D_loss.data[0],
                                                            G_loss.data[0]))
            if i == 0 or i == len(data_loader):

                samples = G(z).data.numpy()[:16]

                fig = plt.figure(figsize=(4,4))
                gs = gridspec.GridSpec(4,4)
                gs.update(wspace=0.05,hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28,28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}.png'.format(str(c).zfill(3)),bbox_inches='tight')
                c +=1
                plt.close(fig)

