import torch
from torch import nn
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.1):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class PMVAE(nn.Module):
    """
    Polymodal Variational Autoencoder
    """

    def __init__(self, n_approx=1):
        """
        n_aprox - number of gaussians
        """
        super(self.__class__, self).__init__()
        self.n_approx = n_approx

        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.LeakyReLU(inplace=True),
            # (|mu| + |sigma| + |decision|) * num_gaussians
            nn.Linear(400, 3 * 20 * n_approx),
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.LeakyReLU(inplace=True),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        batch_size, _ = x.shape
        mu, logvar, dec = self.encoder(x).chunk(3, dim=1)

        mu = mu.view(-1, 20, self.n_approx)
        logvar = logvar.view(-1, 20, self.n_approx)
        # gumbel softmax over decision matrix rows
        dec = gumbel_softmax(dec.reshape(-1, self.n_approx)).view(-1, 20, self.n_approx)

        # 'choose' one component of every matrix row
        return (mu * dec).sum(2), (logvar * dec).sum(2), dec

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.Tensor.detach(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar, dec = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, dec
