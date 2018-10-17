import  os, math, time
import  torch
from    torch import nn
from    torch.utils.data import DataLoader
from    torchvision import transforms
from    torchvision.datasets import MNIST
from    torchvision.utils import save_image
import  visdom
from    torch.nn import functional as F






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



class AutoEncoder2(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

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


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.vars = nn.ParameterList([
            # encoder fc1
            nn.Parameter(torch.empty(256, 28 * 28)),
            nn.Parameter(torch.empty(256)),
            # encoder fc2
            nn.Parameter(torch.empty(64, 256)),
            nn.Parameter(torch.empty(64)),
            # decoder fc1
            nn.Parameter(torch.empty(256, 64)),
            nn.Parameter(torch.empty(256)),
            # decoder fc2
            nn.Parameter(torch.empty(28 * 28, 256)),
            nn.Parameter(torch.empty(28 * 28)),
        ])

    def weights_init(self):

        idx = 0

        def linear_init(weight, bias):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        # Encoder fc1
        weight,bias = self.vars[idx], self.vars[idx + 1]
        linear_init(weight, bias)
        idx += 2

        # Encoder fc2
        weight,bias = self.vars[idx], self.vars[idx + 1]
        linear_init(weight, bias)
        idx += 2

        # Decoder fc1
        weight,bias = self.vars[idx], self.vars[idx + 1]
        linear_init(weight, bias)
        idx += 2

        # Decoder fc2
        weight,bias = self.vars[idx], self.vars[idx + 1]
        linear_init(weight, bias)
        idx += 2

    def forward(self, x, vars=None):
        """

        :param x: [b, 1, 28, 28]
        :return:
        """
        if vars is None:
            vars = self.vars
        idx = 0

        batchsz, c, imgsz, _ = x.shape
        # [b, 1, 28, 28] => [b, 28 * 28]
        x = x.view(batchsz, -1)

        # Encoder fc1
        x = F.linear(x, vars[idx], vars[idx + 1])
        x = F.relu(x)
        idx += 2

        # Encoder fc2
        x = F.linear(x, vars[idx], vars[idx + 1])
        x = F.relu(x)
        idx += 2

        # Decoder fc1
        x = F.linear(x, vars[idx], vars[idx + 1])
        x = F.relu(x)
        idx += 2

        # Decoder fc2
        x = F.linear(x, vars[idx], vars[idx + 1])
        x = torch.sigmoid(x)
        idx += 2

        return x


    def parameters(self, vars=None):
        if vars is None:
            return self.vars
        else:
            return vars

    def zero_grad(self):
        with torch.no_grad(): # ???
            for p in self.vars:
                if p.grad is not None:
                    p.grad.zero_()

    def extra_repr(self):
        return "MAML Basic Net(Variables:{0})".format(len(self.vars))




def main():
    num_epochs = 200
    meta_task = 8

    update_lr = 0.05
    update_step = 5

    device = torch.device('cuda')
    model = AutoEncoder()
    model.to(device)
    model.weights_init()

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=meta_task * 200, shuffle=True)


    criterion1 = nn.BCELoss().to(device)
    criterion2 = nn.MSELoss().to(device)
    # integrate theta with optimizer
    meta_optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    vis = visdom.Visdom()


    for epoch in range(num_epochs):

        t0 = time.time()

        for img, _ in dataloader:

            img = img.to(device)

            # [b, 1, 28, 28]
            batchsz, c, imgsz, _ = img.shape
            # split into support and query
            img1, img2 = torch.chunk(img, chunks=2, dim=0)
            # divide into multi-tasks
            supports = torch.chunk(img1, chunks=meta_task, dim=0)
            querys = torch.chunk(img2, chunks=meta_task, dim=0)

            losses_q = []
            for support, query in zip(supports, querys):

                # forward on theta
                output = model.forward(support)
                loss1 = criterion1(output, support.view(support.size(0), -1))
                # computer derivate in relation to theta
                grad = torch.autograd.grad(loss1, model.parameters())
                # theta_prime = theta - lr * grad
                fast_weights = list(map(lambda x: x[1] - update_lr * x[0], zip(grad, model.vars)))

                for step in range(1, update_step):

                    output = model.forward(support, fast_weights)
                    loss = criterion1(output, support.view(support.size(0), -1))

                    # computer derivate in relation to theta
                    grad = torch.autograd.grad(loss, fast_weights)
                    # theta_prime = theta - lr * grad
                    fast_weights = list(map(lambda x: x[1] - update_lr * x[0], zip(grad, fast_weights)))

                # compute query loss on theta_prime
                output = model.forward(query, fast_weights)
                loss = criterion1(output, query.view(query.size(0), -1))

                losses_q.append(loss)

            # summarize all loss of query
            loss = torch.stack(losses_q).sum()

            # meta-update
            meta_optim.zero_grad()
            loss.backward()
            meta_optim.step()

        # each epoch
        t1 = time.time()
        output = output.view(output.size(0), 1, 28, 28)
        output = F.upsample(output, scale_factor=3)
        vis.images(output, nrow=10, win='ae_meta', opts={'caption':'meta-autoencoder'})

        print('epoch [{}/{}], BCE loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()), t1 - t0)



    torch.save(model.state_dict(), './sim_autoencoder.pth')


if __name__ == '__main__':
    main()