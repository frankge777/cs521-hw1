import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


## Dataloaders
# train_dataset = datasets.CIFAR10('cifar10_data/', train=True, download=True, transform=transforms.Compose(
#     [transforms.ToTensor()]
# ))
test_dataset = datasets.CIFAR10('cifar10_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def tp_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return .5 * (x + delta) * (1 - ind1) * (1 - ind2) + x * ind2

def tp_smoothed_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return (x + delta) ** 2 / (4 * delta) * (1 - ind1) * (1 - ind2) + x * ind2

class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std

class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs
    
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
        elif self.activation[:6] == '3prelu':
            act = tp_relu(preact, delta=float(self.activation.split('relu')[1]))
        elif self.activation[:8] == '3psmooth':
            act = tp_smoothed_relu(preact, delta=float(self.activation.split('smooth')[1]))
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False,
        activation='relu', fts_before_bn=False, normal='none'):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = 64
        self.avg_preact = None
        self.activation = activation
        self.fts_before_bn = fts_before_bn
        if normal == 'cifar10':
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
            self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
            print('no input normalization')
        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride, self.activation))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if return_features and self.fts_before_bn:
            return out.view(out.size(0), -1)
        out = F.relu(self.bn(out))
        if return_features:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def PreActResNet18(n_cls, cuda=True, half_prec=False, activation='relu', fts_before_bn=False,
    normal='none'):
    #print('initializing PA RN-18 with act {}, normal {}'.format())
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, cuda=cuda, half_prec=half_prec,
        activation=activation, fts_before_bn=fts_before_bn, normal=normal)


# intialize the model
model = PreActResNet18(10, cuda=True, activation='softplus1').to(device)
model.eval()
def pgd_linf_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        # TODO: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        # TODO: compute the adv_x
        # find delta, clamp with eps
        grad = adv_x.grad.data
        adv_x = adv_x.detach() + eps_step * grad.sign()
        # project to l_inf ball around original x
        delta = torch.clamp(adv_x - x, min=-eps, max=eps)
        adv_x = torch.clamp(x + delta, 0.0, 1.0).detach()
        adv_x.requires_grad_(True)
   
    return adv_x

def pgd_l2_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        batch_size = x.size()[0]
        # TODO: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        grad = adv_x.grad.data
        print(grad)
        grad_flat = grad.view(batch_size, -1)
        grad_norm = torch.norm(grad_flat, p=2, dim=1).view(batch_size, 1, 1, 1)
        grad_norm = torch.clamp(grad_norm, min=1e-12)  # avoid div by zero
        normalized_grad = grad / grad_norm
        adv_x = adv_x.detach() + eps_step * normalized_grad
        delta = adv_x - x
        delta_flat = delta.view(batch_size, -1)
        delta_norm = torch.norm(delta_flat, p=2, dim=1)
        delta_norm_clamped = torch.clamp(delta_norm, min=1e-12)
        scale = torch.min(torch.ones_like(delta_norm_clamped), eps / delta_norm_clamped)
        scale = scale.view(batch_size, 1, 1, 1)
        delta = delta * scale
        adv_x = torch.clamp(x + delta, 0.0, 1.0).detach()
        adv_x.requires_grad_(True)



        # TODO: compute the adv_x
        # find delta, clamp with eps, project delta to the l2 ball
        # HINT: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py 
   
    return adv_x

def test_model_on_single_attack(model, attack='pgd_linf', eps=0.1):
    k = 10
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        eps_step = eps / 4.0
        tot_test, tot_correct = 0.0, 0.0
        if attack == 'pgd_linf':
            x_adv = pgd_linf_untargeted(model, x_batch, y_batch,
                                        k=k, eps=eps, eps_step=eps_step)

            # TODO: get x_adv untargeted pgd linf with eps, and eps_step=eps/4
        elif attack == 'pgd_l2':
            x_adv = pgd_l2_untargeted(model, x_batch, y_batch,
                                      k=k, eps=eps, eps_step=eps_step)
            # TODO: get x_adv untargeted pgd l2 with eps, and eps_step=eps/4
        else:
            pass
        
        # get the testing accuracy and update tot_test and tot_acc
        with torch.no_grad():
            outputs = model(x_adv)
            preds = outputs.argmax(dim=1)
            correct = (preds == y_batch).sum().item()
            tot_correct += correct
            tot_test += x_batch.size(0)

        tot_acc += tot_correct / tot_test

            
    print('Robust accuracy %.5lf' % (tot_acc/tot_test), f'on {attack} attack with eps = {eps}')
    
def test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75, k=40, random_start=True):
    """
    Evaluate union (multi-norm) robustness: Δ = B2(eps_l2) ∪ B∞(eps_linf).
    A sample is considered robust under the union if the model's prediction remains correct
    for BOTH the l_inf and the l2 adversarial examples generated (i.e., adversary cannot
    find any adversarial example within either ball that flips the prediction).
    Returns robust accuracy under union threat model.
    """
    model.eval()
    tot_test = 0
    tot_robust = 0

    # choose step sizes
    eps_step_linf = eps_linf / 4.0
    eps_step_l2 = eps_l2 / max(1.0, k)

    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader),
                                              total=len(test_loader),
                                              desc="Evaluating multi-norm (union)"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # generate adversarial examples (independently)
        x_adv_linf = pgd_linf_untargeted(model, x_batch, y_batch, k=k, eps=eps_linf, eps_step=eps_step_linf)
        x_adv_l2   = pgd_l2_untargeted(model, x_batch, y_batch, k=k, eps=eps_l2, eps_step=eps_step_l2)

        # get predictions
        with torch.no_grad():
            out_linf = model(x_adv_linf)
            pred_linf = out_linf.argmax(dim=1)
            out_l2 = model(x_adv_l2)
            pred_l2 = out_l2.argmax(dim=1)

            # a sample is robust under the union iff both adversarial predictions equal the true label
            robust_mask = (pred_linf == y_batch) & (pred_l2 == y_batch)
            tot_robust += robust_mask.sum().item()
            tot_test += x_batch.size(0)

    robust_acc = tot_robust / tot_test
    print(f'Robust accuracy {robust_acc:.5f} under union (B2 ∪ B∞) with eps_linf={eps_linf}, eps_l2={eps_l2}')
    return robust_acc



model.load_state_dict(torch.load('models/pretr_RAMP.pth'))
# Evaluate on multi attacks with model 3
model.to(device)
test_model_on_multi_attacks(model, eps_linf=8/255.0, eps_l2=0.75, k=10)
print("multi_norm pretr_RAMP model")
