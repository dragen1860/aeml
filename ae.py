import  os, time
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch.utils.data import DataLoader
from    torchvision import transforms
from    torchvision.datasets import MNIST
from    torchvision.utils import save_image
import  visdom

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 200
batch_size = 128
learning_rate = 1e-3


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


img_transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x:min_max_normalization(x, 0, 1),
    lambda x:torch.round(x)
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




def main():
    device = torch.device('cuda')
    model = autoencoder()
    model.to(device)

    criterion1 = nn.BCELoss().to(device)
    criterion2 = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    vis = visdom.Visdom()

    for epoch in range(num_epochs):

        t0 = time.time()

        for img, _ in dataloader:
            # [b, 1, 28, 28] => [b, 784]
            img = img.view(img.size(0), -1).to(device)
            #
            output = model(img)
            loss1 = criterion1(output, img)
            loss2 = criterion2(output, img)
            #
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()


        time.sleep(1.3)
        t1 = time.time()

        output = output.view(output.size(0), 1, 28, 28)
        output = F.upsample(output, scale_factor=3)
        vis.images(output, nrow=12, win='ae', opts={'caption':'autoencoder'})

        print('epoch [{}/{}], BCE loss:{:.4f}, MSE loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss1.item(), loss2.item()), t1-t0)



    torch.save(model.state_dict(), './sim_autoencoder.pth')


if __name__ == '__main__':
    main()