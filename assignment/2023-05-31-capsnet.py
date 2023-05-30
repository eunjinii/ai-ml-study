import torch
from torch import nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from capsnet import CapsNet, CapsuleLoss
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler
import torch.cuda.nvtx as nvtx

# Check cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def squash(x, dim=-1):
    nvtx.range_push("Squash")
    """ return the squash function of x. """
    
    #squash0_cuda.forward(x)
    
        
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    squash = scale * x / (squared_norm.sqrt() + 1e-8)

    nvtx.range_pop()
    return x

class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: out_capsules * (batch_size, out_channels, height, weight)
        nvtx.range_push("PConv")
        out = self.conv(x)
        nvtx.range_pop()
        # Flatten out: (batch_size, out_capsules * height * weight, out_channels)
        nvtx.range_push("Pflatten")
        batch_size = out.shape[0]
        #with profiler.record_function("squash1"):
        out = out.contiguous().view(batch_size, -1, self.out_channels)
        nvtx.range_pop()
        return squash(out)


class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        """
        Initialize the layer.
        Args:
            in_dim: 		Dimensionality of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            out_caps: 		Number of capsules in the capsule layer
            out_dim: 		Dimensionality, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dims, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, out_caps, in_caps, out_dim)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, out_dim)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along out_dim
            #with profiler.record_function("squash2"):
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_caps, in_caps, out_dim) @ (batch_size, out_caps, out_dim, 1)
            # -> (batch_size, out_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along out_dim
        #with profiler.record_function("squash3"):
        v = squash(s)
        #v = s.clone()
        #squash0_cuda.forward(v)
        
        return v


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer1
        self.conv1 = nn.Conv2d(3, 16, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
       
       # Conv2d layer2
        self.conv2 = nn.Conv2d(16, 64, 9, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

       # Conv2d layer3
        self.conv3 = nn.Conv2d(64, 256, 9, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)


        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=32 * 6 * 6 ,
                                    out_caps=1000,
                                    out_dim=16,
                                    num_routing=3)


        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1600),
            nn.ReLU(inplace=True),
            nn.Linear(1600, 3072),
            nn.Sigmoid())

    def forward(self, x):
                
        nvtx.range_push("Conv")
        out = self.bn1(self.conv1(x))
        nvtx.range_pop()
        nvtx.range_push("Relu")
        out = self.relu1(out)
        nvtx.range_pop()
        nvtx.range_push("Conv")
        out = self.bn2(self.conv2(out))
        nvtx.range_pop()
        nvtx.range_push("Relu")
        out = self.relu2(out)
        nvtx.range_pop()
        nvtx.range_push("Conv")
        out = self.bn3(self.conv3(out))
        nvtx.range_pop()
        nvtx.range_push("Relu")
        out = self.relu3(out)
        nvtx.range_pop()
        nvtx.range_push("Primary Caps")
        out = self.primary_caps(out)
        nvtx.range_pop()
        nvtx.range_push("Digit Caps")
        out = self.digit_caps(out)
        nvtx.range_pop()
        # Shape of logits: (batch_size, out_capsules)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(1000).to(device).index_select(dim=0, index=torch.argmax(logits, dim=1))

        # Reconstruction
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1))

        return logits, reconstruction


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-4
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        # Reconstruction loss
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)
        self.reconstruction_loss_scalar /= 1.001

        # Combine two losses
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss