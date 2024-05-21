import argparse
from os.path import join

import torch

from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets import get_data_iter
from net.style_transfer import StyleTransfer, Fusion, get_encoder, get_decoder, adjust_learning_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_data_dir', type=str, default='/home/zg34/datasets/COCO/train',
                        help='Path to content images')
    parser.add_argument('--style_data_dir', type=str, default='/home/zg34/datasets/PainterByNumber',
                        help='Path to style images')
    parser.add_argument('--vgg_model_path', type=str, default=r'./net/vgg_normalised.pth',
                        help='Path to vgg19 pretrained model')
    parser.add_argument('--log_dir', type=str, default='./logs/debug',
                        help='Path to save logs')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--style_weight', type=float, default=3.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--id1_weight', type=float, default=50.0)
    parser.add_argument('--id2_weight', type=float, default=1.0)
    parser.add_argument('--save_model_interval', type=int, default=10000,
                        help='step interval for saving model')
    parser.add_argument('--show_img_interval', type=int, default=1000,
                        help='step interval for showing images')
    parser.add_argument('--random_seed', type=int, default=3407,
                        help='manual seed for PyTorch,'
                             'torch.manual_seed(3407) is all you need:'
                             'https://arxiv.org/abs/2109.08203')
    args = parser.parse_args()
    args.save_dir = join(args.log_dir, 'checkpoints')
    for arg in vars(args):
        print('{}={}'.format(arg, getattr(args, arg)))
    # create folders and tensorboard writer
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    # initialize torch and device
    torch.manual_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_name = torch.cuda.get_device_name(device)

    # initialize network, data, optimizer
    net = StyleTransfer(Fusion(512), get_encoder(args.vgg_model_path), get_decoder(None)).to(device)

    content_iter = get_data_iter(device=device,
                                 data_dir=args.content_data_dir,
                                 batch_size=args.batch_size,
                                 num_workers=0)
    style_iter = get_data_iter(device=device,
                               data_dir=args.style_data_dir,
                               batch_size=args.batch_size,
                               num_workers=0)

    optimizer = torch.optim.Adam([{'params': net.decoder.parameters()},
                                  {'params': net.fusion.parameters()}],
                                 lr=args.lr)

    pbar = tqdm(total=args.max_iter)
    for i in range(args.max_iter):
        adjust_learning_rate(optimizer, iteration_count=i, start_lr=args.lr, lr_decay=args.lr_decay)
        content_images = next(content_iter)
        style_images = next(style_iter)
        nst_images, loss_c, loss_s, loss_id_1, loss_id_2 = net(content=content_images, style=style_images)
        loss = (args.content_weight * loss_c +
                args.style_weight * loss_s +
                args.id1_weight * loss_id_1 +
                args.id2_weight * loss_id_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(
            'l_c={:.4f},l_s={:.4f},l_id1={:.4f},l_id2={:.4f}'.format(
                loss_c.item(), loss_s.item(), loss_id_1.item(), loss_id_2.item())
        )
        writer.add_scalar('content loss', loss_c.item(), i + 1)
        writer.add_scalar('style loss', loss_s.item(), i + 1)
        writer.add_scalar('identity loss 1', loss_id_1.item(), i + 1)
        writer.add_scalar('identity loss 2', loss_id_2.item(), i + 1)
        writer.add_scalar('total loss', loss.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            to_save = {'fusion': net.fusion.state_dict(),
                       'decoder': net.decoder.state_dict()}
            torch.save(to_save, save_dir / 'checkpoints_iter_{:d}.pth'.format(i + 1))
        if (i + 1) % args.show_img_interval == 0 or (i + 1) == args.max_iter or i == 0:
            with torch.no_grad():
                writer.add_images('content/style/NST',
                                  torch.concat([content_images.detach(),
                                                style_images.detach(),
                                                nst_images.detach()],
                                               dim=2),
                                  i + 1)
        pbar.update(1)
    pbar.close()
    writer.close()
    print('{} finished training.'.format(GPU_name))
