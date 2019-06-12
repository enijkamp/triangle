import os
import random

import shutil
import datetime
import logging
import sys
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.utils.data


def get_args(exp_id):
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', default=exp_id)

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nez', type=int, default=1, help='size of the output of ebm')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nif', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3, help='number of channels')

    parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')

    parser.add_argument('--niter', type=int, default=1500, help='number of epochs to train for')
    parser.add_argument('--e_lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--g_lr', type=float, default=0.0003, help='learning rate, default=0.0002')
    parser.add_argument('--i_lr', type=float, default=0.0003, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--vfactor', type=float, default=1.0, help='factor for vae component')

    parser.add_argument('--is_grad_clampE', type=bool, default=True, help='whether doing the gradient clamp for E')
    parser.add_argument('--max_normE', type=float, default=100, help='max norm allowed for E')

    parser.add_argument('--is_grad_clampG', type=bool, default=True, help='whether doing the gradient clamp for G')
    parser.add_argument('--max_normG', type=float, default=100, help='max norm allowed for G')

    parser.add_argument('--is_grad_clampI', type=bool, default=True, help='whether doing the gradient clamp for I')
    parser.add_argument('--max_normI', type=float, default=100, help='max norm allowed for I')

    parser.add_argument('--e_decay', type=float, default=0.0000, help='weight decay for EBM')
    parser.add_argument('--i_decay', type=float, default=0.0000, help='weight decay for I')
    parser.add_argument('--g_decay', type=float, default=0.0005, help='weight decay for G')

    parser.add_argument('--e_gamma', type=float, default=0.998, help='lr exp decay for EBM')
    parser.add_argument('--i_gamma', type=float, default=0.998, help='lr exp decay for I')
    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr exp decay for G')

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--netI', default='', help="path to netI (to continue training)")

    return parser.parse_args()


def train(device, args, output_dir, logger):

    # output
    outf_recon = output_dir + '/recon'
    outf_syn = output_dir + '/syn'
    outf_test = output_dir + '/test'
    outf_ckpt = output_dir + '/ckpt'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(outf_recon, exist_ok=True)
    os.makedirs(outf_syn, exist_ok=True)
    os.makedirs(outf_test, exist_ok=True)
    os.makedirs(outf_ckpt, exist_ok=True)

    # data
    dataset = datasets.CIFAR10(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))
    dataset_full = np.array([x[0].cpu().numpy() for x in iter(dataset)])
    unnormalize = lambda img: img / 2.0 + 0.5

    # params
    nz = int(args.nz)
    nez = int(args.nez)
    ngf = int(args.ngf)
    ndf = int(args.ndf)
    nif = int(args.nif)
    nc = int(args.nc)

    # tensorflow
    def create_lazy_session():
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        return tf.Session(config=config)

    import inception_score_v2_tf as is_v2
    import fid_v2_tf as fid_v2

    # models
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train_flag():
        netG.train()
        netI.train()
        netE.train()

    class _netG(nn.Module):
        def __init__(self, nz, nc, ngf):
            super(_netG, self).__init__()

            self.deconv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
            self.deconv1_bn = nn.BatchNorm2d(ngf * 8)

            self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
            self.deconv2_bn = nn.BatchNorm2d(ngf * 8)

            self.deconv3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            self.deconv3_bn = nn.BatchNorm2d(ngf * 4)

            self.deconv4 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            self.deconv4_bn = nn.BatchNorm2d(ngf * 2)

            self.deconv5 = nn.ConvTranspose2d(ngf * 2, nc, 3, 1, 1)

        def forward(self, input):
            oG_l1 = F.relu(self.deconv1_bn(self.deconv1(input)))
            oG_l2 = F.relu(self.deconv2_bn(self.deconv2(oG_l1)))
            oG_l3 = F.relu(self.deconv3_bn(self.deconv3(oG_l2)))
            oG_l4 = F.relu(self.deconv4_bn(self.deconv4(oG_l3)))
            oG_out = torch.tanh(self.deconv5(oG_l4))
            return oG_out

    class _netE(nn.Module):
        def __init__(self, nc, nez, ndf):
            super(_netE, self).__init__()

            self.conv1 = nn.Conv2d(nc, ndf, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            self.conv4 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
            self.conv5 = nn.Conv2d(ndf * 4, nez, 4, 1, 0)

        def forward(self, input):
            oE_l1 = F.leaky_relu(self.conv1(input), 0.2)
            oE_l2 = F.leaky_relu(self.conv2(oE_l1), 0.2)
            oE_l3 = F.leaky_relu(self.conv3(oE_l2), 0.2)
            oE_l4 = F.leaky_relu(self.conv4(oE_l3), 0.2)
            oE_out = self.conv5(oE_l4)
            return oE_out

    class _netI(nn.Module):
        def __init__(self, nc, nz, nif):
            super(_netI, self).__init__()

            self.conv1 = nn.Conv2d(nc, nif, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=False)
            self.conv4 = nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=False)
            self.conv51 = nn.Conv2d(nif * 8, nz, 4, 1, 0)  # for mu
            self.conv52 = nn.Conv2d(nif * 8, nz, 4, 1, 0)  # for log_sigma

        def forward(self, input):
            oI_l1 = F.leaky_relu(self.conv1(input), 0.2)
            oI_l2 = F.leaky_relu(self.conv2(oI_l1), 0.2)
            oI_l3 = F.leaky_relu(self.conv3(oI_l2), 0.2)
            oI_l4 = F.leaky_relu(self.conv4(oI_l3), 0.2)

            oI_mu = self.conv51(oI_l4)
            oI_log_sigma = self.conv52(oI_l4)
            return oI_mu, oI_log_sigma

    netG = _netG(nz, nc, ndf).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))

    netE = _netE(nc, nez, ngf).to(device)
    netE.apply(weights_init)
    if args.netE != '':
        netE.load_state_dict(torch.load(args.netE))

    netI = _netI(nc, nz, nif).to(device)
    netI.apply(weights_init)
    if args.netI != '':
        netI.load_state_dict(torch.load(args.netI))

    input = torch.FloatTensor(args.batchSize, nc, args.imageSize, args.imageSize).to(device)
    noise = torch.FloatTensor(args.batchSize, nz, 1, 1).to(device)
    fixed_noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_().to(device)
    fixed_noiseV = Variable(fixed_noise)
    mse_loss = nn.MSELoss(reduction='sum').to(device)

    def compute_energy(disc_score):
        if args.energy_form == 'tanh':
            energy = torch.tanh(-disc_score.squeeze())
        elif args.energy_form == 'sigmoid':
            energy = F.sigmoid(-disc_score.squeeze())
        elif args.energy_form == 'identity':
            energy = disc_score.squeeze()
        elif args.energy_form == 'softplus':
            energy = F.softplus(-disc_score.squeeze())
        return energy

    def diag_normal_NLL(z, z_mu, z_log_sigma):
        # define the Negative Log Probability of Normal which has diagonal cov
        # input: [batch nz, 1, 1] squeeze it to batch nz
        # return: shape is [batch]
        nll = 0.5 * torch.sum(z_log_sigma.squeeze(), dim=1) + \
              0.5 * torch.sum((torch.mul(z - z_mu, z - z_mu) / (1e-6 + torch.exp(z_log_sigma))).squeeze(), dim=1)
        return nll.squeeze()

    def reparametrize(mu, log_sigma, is_train=True):
        if is_train:
            std = torch.exp(log_sigma.mul(0.5))
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    optimizerE = optim.Adam(netE.parameters(), lr=args.e_lr, betas=(args.beta1, 0.999), weight_decay=args.e_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.beta1, 0.999), weight_decay=args.g_decay)
    optimizerI = optim.Adam(netI.parameters(), lr=args.i_lr, betas=(args.beta1, 0.999), weight_decay=args.i_decay)

    lrE_schedule = optim.lr_scheduler.ExponentialLR(optimizerE, args.e_gamma)
    lrG_schedule = optim.lr_scheduler.ExponentialLR(optimizerG, args.g_gamma)
    lrI_schedule = optim.lr_scheduler.ExponentialLR(optimizerI, args.i_gamma)

    stats_headings = [['epoch', '{:>14}', '{:>14d}'],
                      ['errRecon', '{:>14}', '{:>14.3f}'],
                      ['errLatent', '{:>14}', '{:>14.3f}'],
                      ['E_T', '{:>14}', '{:>14.3f}'],
                      ['E_F', '{:>14}', '{:>14.3f}'],
                      ['err(I)', '{:>14}', '{:>14.3f}'],
                      ['err(G)', '{:>14}', '{:>14.3f}'],
                      ['err(E)', '{:>14}', '{:>14.3f}'],
                      ['KLD(z)', '{:>14}', '{:>14.3f}'],
                      ['lr(E)', '{:>14}', '{:>14.6f}'],
                      ['lr(G)', '{:>14}', '{:>14.6f}'],
                      ['lr(I)', '{:>14}', '{:>14.6f}'],
                      ['inc_v2', '{:>14}', '{:>14.3f}'],
                      ['fid_v2', '{:>14}', '{:>14.3f}'],
                      ]

    logger.info(' ')
    logger.info(''.join([h[1] for h in stats_headings]).format(*[h[0] for h in stats_headings]))

    is_v2_score, fid_v2_score = 0., 0.

    num_samples = 50000
    noise_z = torch.FloatTensor(100, nz, 1, 1)
    new_noise = lambda: noise_z.normal_().cuda()

    for epoch in range(args.niter):

        stats_values = {k[0]: 0 for k in stats_headings}
        stats_values['epoch'] = epoch

        lrE_schedule.step()
        lrI_schedule.step()
        lrG_schedule.step()

        num_batch = len(dataloader.dataset) / args.batchSize
        for i, data in enumerate(dataloader, 0):

            train_flag()

            """
            Train EBM 
            """
            netE.zero_grad()
            real_cpu, _ = data
            real_cpu = real_cpu.to(device)
            batch_size = real_cpu.size(0)
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputV = Variable(input)
            disc_score_T = netE(inputV)
            Eng_T = compute_energy(disc_score_T)
            E_T = torch.mean(Eng_T)
            noise.resize_(batch_size, nz, 1, 1).normal_()  # or Uniform
            noiseV = Variable(noise)
            samples = netG(noiseV)
            disc_score_F = netE(samples.detach())
            Eng_F = compute_energy(disc_score_F)
            E_F = torch.mean(Eng_F)
            errE = E_T - E_F
            errE.backward()
            if args.is_grad_clampE:
                torch.nn.utils.clip_grad_norm_(netE.parameters(), args.max_normE)
            optimizerE.step()

            """
            Train I
            besides the original acd1 which is only build on the generated data, we also consider build on true data
            (1) reconstruct of train data
            (2) kld with prior
            (3) reconstruct of latent codes given generated data
            """
            netI.zero_grad()

            # part 1: reconstruction on train data (May get per-batch loss)
            infer_z_mu_true, infer_z_log_sigma_true = netI(inputV)
            z_input = reparametrize(infer_z_mu_true, infer_z_log_sigma_true)
            inputV_recon = netG(z_input)
            errRecon = mse_loss(inputV_recon, inputV) / batch_size
            errKld = -0.5 * torch.mean(1 + infer_z_log_sigma_true - infer_z_mu_true.pow(2) - infer_z_log_sigma_true.exp())

            # part 3: reconstruction on latent z based on the generated data
            infer_z_mu_gen, infer_z_log_sigma_gen = netI(samples.detach())
            errLatent = 0.1 * torch.mean(diag_normal_NLL(noiseV, infer_z_mu_gen, infer_z_log_sigma_gen))

            errI = args.vfactor * (errRecon + errKld) + errLatent
            errI.backward()
            if args.is_grad_clampI:
                torch.nn.utils.clip_grad_norm_(netI.parameters(), args.max_normI)
            optimizerI.step()

            """
            Train G
            besides the original acd1, we add vae criterion which pushes the generator to cover data
            (1) reconstruct the train given re-parameterized z
            (2) MLE of energy and inference: 
                (a) reconstruct the latent space
                (b) Fool the energy discriminator
            """
            netG.zero_grad()
            # part 1: reconstruct the train data
            infer_z_mu_true, infer_z_log_sigma_true = netI(inputV)
            z_input = reparametrize(infer_z_mu_true, infer_z_log_sigma_true)
            inputV_recon = netG(z_input)
            errRecon = mse_loss(inputV_recon, inputV) / batch_size

            # part2: (b): fool discriminator
            disc_score_F = netE(samples)
            Eng_F = compute_energy(disc_score_F)
            E_F = torch.mean(Eng_F)

            # part2: (a) : reconstruct the latent space
            infer_z_mu_gen, infer_z_log_sigma_gen = netI(samples)
            errLatent = 0.1 * torch.mean(diag_normal_NLL(noiseV, infer_z_mu_gen, infer_z_log_sigma_gen))

            errG = args.vfactor * errRecon + E_F + errLatent
            errG.backward()
            if args.is_grad_clampG:
                torch.nn.utils.clip_grad_norm_(netG.parameters(), args.max_normG)
            optimizerG.step()

            stats_values['errRecon'] += errRecon.data.item() / num_batch
            stats_values['errLatent'] += errLatent.data.item() / num_batch
            stats_values['E_T'] += E_T.data.item() / num_batch
            stats_values['E_F'] += E_F.data.item() / num_batch
            stats_values['err(I)'] += errI.data.item() / num_batch
            stats_values['err(G)'] += errG.data.item() / num_batch
            stats_values['err(E)'] += errE.data.item() / num_batch
            stats_values['KLD(z)'] += errKld.data.item() / num_batch

        # images
        if epoch % 10 == 0 or epoch == (args.niter - 1):
            gen_samples = netG(fixed_noiseV)
            vutils.save_image(gen_samples.data, '%s/epoch_%03d_samples.png' % (outf_syn, epoch), normalize=True, nrow=10)

            infer_z_mu_input, _ = netI(inputV)
            recon_input = netG(infer_z_mu_input)
            vutils.save_image(recon_input.data, '%s/epoch_%03d_reconstruct_input.png' % (outf_recon, epoch), normalize=True, nrow=10)

            infer_z_mu_sample, _ = netI(gen_samples)
            recon_sample = netG(infer_z_mu_sample)
            vutils.save_image(recon_sample.data, '%s/epoch_%03d_reconstruct_samples.png' % (outf_syn, epoch), normalize=True, nrow=10)

            # interpolation
            between_input_list = [inputV[0].data.cpu().numpy()[np.newaxis, ...]]
            zfrom = infer_z_mu_input[0].data.cpu()
            zto = infer_z_mu_input[1].data.cpu()
            fromto = zto - zfrom
            for alpha in np.linspace(0, 1, 8):
                between_z = zfrom + alpha * fromto
                recon_between = netG(Variable(between_z.unsqueeze(0).to(device)))
                between_input_list.append(recon_between.data.cpu().numpy())
            between_input_list.append(inputV[1].data.cpu().numpy()[np.newaxis, ...])
            between_canvas_np = np.concatenate(between_input_list, axis=0)
            vutils.save_image(torch.from_numpy(between_canvas_np), '%s/epoch_%03d_interpolate.png' % (outf_syn, epoch), normalize=True, nrow=10, padding=5)

        # metrics
        if epoch > 0 and (epoch % 50 == 0 or epoch == (args.niter - 1)):
            torch.save(netG.state_dict(), outf_ckpt + '/netG_%03d.pth' % epoch)
            torch.save(netI.state_dict(), outf_ckpt + '/netI_%03d.pth' % epoch)
            torch.save(netE.state_dict(), outf_ckpt + '/netE_%03d.pth' % epoch)

            to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))

            gen_samples = torch.cat([netG(new_noise()).detach().cpu() for _ in range(int(num_samples / 100))])
            gen_samples_np = 255 * unnormalize(gen_samples.numpy())
            gen_samples_np = to_nhwc(gen_samples_np)
            gen_samples_list = [gen_samples_np[i, :, :, :] for i in range(num_samples)]

            is_v2_score = is_v2.inception_score(create_lazy_session, gen_samples_list, resize=True, splits=1)[0]
            fid_v2_score = fid_v2.fid_score(create_lazy_session, 255 * to_nhwc(unnormalize(dataset_full)), gen_samples_np)

        # stats
        stats_values['inc_v2'] = is_v2_score
        stats_values['fid_v2'] = fid_v2_score

        stats_values['lr(G)'] = optimizerG.param_groups[0]['lr']
        stats_values['lr(E)'] = optimizerE.param_groups[0]['lr']
        stats_values['lr(I)'] = optimizerI.param_groups[0]['lr']

        logger.info(''.join([h[2] for h in stats_headings]).format(*[stats_values[k[0]] for k in stats_headings]))

    logger.info('done')


def set_seed(seed):
    assert seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_gpu(device):
    torch.cuda.set_device(device)


def set_cudnn():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def setup_logging(name='main', output_dir='.', console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_exp_id():
    return os.path.splitext(os.path.basename(__file__))[0]


def main():
    # preamble
    exp_id = get_exp_id()
    args = get_args(exp_id)
    output_dir = get_output_dir(exp_id)
    logger = setup_logging(output_dir=output_dir)
    logger.info(args)
    copy_source(__file__, output_dir)

    # device
    device = torch.device(args.device)
    set_gpu(device)
    set_cudnn()
    set_seed(args.seed)

    # go
    train(device, args, output_dir, logger)


if __name__ == '__main__':
    main()


