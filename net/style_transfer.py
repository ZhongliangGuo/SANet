from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def adjust_learning_rate(optimizer, iteration_count, start_lr, lr_decay):
    lr = start_lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calc_mean_std(tensor: Tensor, eps=1e-5):
    mean = tensor.mean(dim=[2, 3], keepdim=True)
    std = (tensor.var(dim=[2, 3], keepdim=True) + eps).sqrt()
    return mean, std


def normalize(tensor: Tensor, eps=1e-5):
    mean, std = calc_mean_std(tensor, eps)
    return (tensor - mean) / std


def get_encoder(pth_path):
    # region vgg define
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    # endregion
    if pth_path:
        vgg.load_state_dict(torch.load(pth_path))
        print('loaded the pth file for vgg')
    return vgg


def get_decoder(pth_path):
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
    if pth_path:
        decoder.load_state_dict(torch.load(pth_path))
        print('loaded the pth file for vgg')
    return decoder


class SANet(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.f = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, content: Tensor, style: Tensor):
        assert content.shape == style.shape
        B, C, H, W = content.shape
        return self.out(torch.bmm(self.h(style).flatten(2, -1),
                                  torch.softmax(torch.bmm(self.f(normalize(content)).flatten(2, -1).permute(0, 2, 1),
                                                          self.f(normalize(style)).flatten(2, -1)),
                                                dim=-1).permute(0, 2, 1)).view(B, C, H, W))


class Fusion(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.sanet4_1 = SANet(in_channels)
        self.sanet5_1 = SANet(in_channels)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                   nn.Conv2d(in_channels, in_channels, 3))

    def forward(self, content_feats: List[Tensor], style_feats: List[Tensor]):
        return self.merge(
            self.sanet4_1(content_feats[-2], style_feats[-2]) +
            self.upsample5_1(self.sanet5_1(content_feats[-1], style_feats[-1]))
        )


class StyleTransfer(nn.Module):
    def __init__(self, fusion: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.fusion = fusion
        self.encoder = encoder
        self.decoder = decoder
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def encode_with_intermediate(self, tensor: Tensor):
        results = [tensor]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, tensor: Tensor):
        for i in range(5):
            tensor = getattr(self, 'enc_{:d}'.format(i + 1))(tensor)
        return tensor

    def calc_content_loss_once(self, pred: Tensor, target: Tensor):
        assert (pred.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(pred, target)

    def calc_content_loss(self, preds: List[Tensor], targets: List[Tensor]):
        assert len(preds) == len(targets)
        loss_s = self.calc_content_loss_once(preds[0], targets[0])
        for i in range(1, 5):
            loss_s += self.calc_content_loss_once(preds[i], targets[i])
        return loss_s

    def calc_style_loss_once(self, pred: Tensor, target: Tensor):
        assert (pred.size() == target.size())
        # assert (target.requires_grad is False) # first, make sure which one requires gradients and which one does not.
        pred_mean, pred_std = calc_mean_std(pred)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(pred_mean, target_mean) + self.mse_loss(pred_std, target_std)

    def calc_style_loss(self, preds: List[Tensor], targets: List[Tensor]):
        assert len(preds) == len(targets)
        loss_s = self.calc_style_loss_once(preds[0], targets[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss_once(preds[i], targets[i])
        return loss_s

    @torch.no_grad()
    def inference(self, content: Tensor, style: Tensor):
        content_feats = self.encode_with_intermediate(content)
        style_feats = self.encode_with_intermediate(style)
        gt = torch.clamp(self.decoder(self.fusion(content_feats, style_feats)), 0, 1)
        return gt

    def forward(self, content: Tensor, style: Tensor):
        content_feats = self.encode_with_intermediate(content)
        style_feats = self.encode_with_intermediate(style)
        gt = torch.clamp(self.decoder(self.fusion(content_feats, style_feats)), 0, 1)
        gt_feats = self.encode_with_intermediate(gt)
        cc = self.decoder(self.fusion(content_feats, content_feats))
        ss = self.decoder(self.fusion(style_feats, style_feats))
        cc_feats = self.encode_with_intermediate(cc)
        ss_feats = self.encode_with_intermediate(ss)
        # calculate loss
        loss_c = (self.calc_content_loss_once(gt_feats[-2], normalize(content_feats[-2])) +
                  self.calc_content_loss_once(gt_feats[-1], normalize(content_feats[-1])))
        loss_s = self.calc_style_loss(gt_feats, style_feats)
        loss_id_1 = self.calc_content_loss_once(cc, content) + self.calc_content_loss_once(ss, style)
        loss_id_2 = self.calc_content_loss(cc_feats, content_feats) + self.calc_content_loss(ss_feats, style_feats)
        return gt, loss_c, loss_s, loss_id_1, loss_id_2
