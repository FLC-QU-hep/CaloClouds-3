import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *
from .flow import *
from .encoders.epic_encoder_cond import EPiC_encoder_cond
from utils.misc import *

import k_diffusion as K


class epicVAE_nFlow_kDiffusion(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = EPiC_encoder_cond(args.latent_dim, input_dim=args.features, cond_features=args.cond_features)
        self.flow = get_flow_model(args)

        if args.dropout_rate == 0.0:
            net = PointwiseNet_kDiffusion(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual)
        else:
            net = PointwiseNet_kDiffusion_Dropout(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual, dropout_rate=args.dropout_rate)
        self.diffusion = Denoiser(net, sigma_data = args.model['sigma_data'], device=args.device)
        # self.diffusion = K.config.make_denoiser_wrapper(args.__dict__)(net)
        # self.diffusion = DiffusionPoint(
        #     net = PointwiseNet_kDiffusion(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual),
        #     var_sched = VarianceSchedule(
        #         num_steps=args.num_steps,
        #         beta_1=args.beta_1,
        #         beta_T=args.beta_T,
        #         mode=args.sched_mode
        #     ))
            
        self.kld = KLDloss()

    def get_loss(self, x, noise, sigma, cond_feats, kl_weight, writer=None, it=None, kld_min=0.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            noise: Noise point cloud (B, N, d).
            sigma: Time (B, ).
            cond_feats: conditioning features (B, C)
        """
        # batch_size, _, _ = x.size()
        # VAE encoder
        z_mu, z_sigma = self.encoder(x, cond_feats)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # VAE-like loss for encoder
        loss_kld = self.kld(mu=z_mu, logvar=z_sigma)
        loss_kld_clamped = torch.clamp(loss_kld, min=kld_min)

        # P(z), Prior probability, parameterized by the flow: z -> w.
        nll = - self.flow.log_prob(z.detach().clone(), cond_feats)    # detach from computational graph if optimizing encoder+diffuison seperate from flow
        loss_prior = nll.mean()

        # Diffusion MSE loss: Negative ELBO of P(X|z)
        z = torch.cat([z, cond_feats], -1)
        #loss_diffusion = self.diffusion.get_loss(x, z)    # diffusion loss
        loss_diffusion = self.diffusion.loss(x, noise, sigma, context=z).mean()    # diffusion loss
        # print('max values: ', loss_diffusion.max().item(), loss_diffusion.min().item())
        # print('shape of loss_diffusion: ', loss_diffusion.shape)

        # Total loss
        loss = kl_weight*(loss_kld_clamped) + loss_diffusion
        # print(loss_kld_clamped.item(), loss_recons.item())

        # if writer is not None:
        #     # writer.log_metric('train/loss_e', loss_entropy, it)
        #     writer.log_metric('train/loss_kld', loss_kld, it)
        #     writer.log_metric('train/loss_kld_clamped', loss_kld_clamped, it)
        #     writer.log_metric('train/loss_prior', loss_prior, it)
        #     writer.log_metric('train/loss_diffusion', loss_diffusion, it)
        #     writer.log_metric('train/z_mean', z_mu.mean(), it)
        #     writer.log_metric('train/z_mag', z_mu.abs().max(), it)
        #     writer.log_metric('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss, loss_prior


    def sample(self, cond_feats, num_points, config):
        batch_size, _ = cond_feats.size()

        # contect / latent space
        z = self.flow.sample(context=cond_feats, num_samples=1).view(batch_size, -1)  # B,F
        z = torch.cat([z, cond_feats], -1)   # B, F+C

        sigmas = K.sampling.get_sigmas_karras(config.num_steps, config.sigma_min, config.sigma_max, rho=7., device=z.device)

        x_T = torch.randn([z.size(0), num_points, config.features], device=z.device) * config.sigma_max

        if config.sampler == 'euler':
            x_0 = K.sampling.sample_euler(self.diffusion, x_T, sigmas, extra_args={'context' : z})
        elif config.sampler == 'heun':
            x_0 = K.sampling.sample_heun(self.diffusion, x_T, sigmas, extra_args={'context' : z}, s_churn=config.s_churn, s_noise=config.s_noise)
        elif config.sampler == 'dpmpp_2m':
            x_0 = K.sampling.sample_dpmpp_2m(self.diffusion, x_T, sigmas, extra_args={'context' : z})
        elif config.sampler == 'dpmpp_2s_ancestral':
            x_0 = K.sampling.sample_dpmpp_2s_ancestral(self.diffusion, x_T, sigmas, extra_args={'context' : z})
        elif config.sampler == 'sample_euler_ancestral':
            x_0 = K.sampling.sample_euler_ancestral(self.diffusion, x_T, sigmas, extra_args={'context' : z})
        elif config.sampler == 'sample_lms':
            x_0 = K.sampling.sample_lms(self.diffusion, x_T, sigmas, extra_args={'context' : z})
        elif config.sampler == 'sample_dpmpp_2m_sde':
            x_0 = K.sampling.sample_dpmpp_2m_sde(self.diffusion, x_T, sigmas, extra_args={'context' : z})
        else:
            raise NotImplementedError('Sampler not implemented')
        return x_0
    
    # def inference(self, num_points, cfg, num_samples=256):
    #     with torch.no_grad():
    #         z = torch.randn([num_samples, cfg.latent_dim]).to(cfg.device)
    #         x = self.sample(z, num_points, flexibility=cfg.flexibility)
    #     return x

    # def sample_fromData(self, x, cond_feats, num_points, flexibility):
    #     z_mu, z_sigma = self.encoder(x, cond_feats)
    #     z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
    #     z = torch.cat([z, cond_feats], -1)   # B,F+C
    #     samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
    #     return samples


    def get_loss(self, x, cond_feats, model_teacher):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            sigma: Time (B, ).
            model_teacher: teacher model as score function for ODE solver
        """

        loss = None

        return loss
    


# from: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py#L12
class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1., device='cuda'):
        super().__init__()
        self.inner_model = inner_model
        if isinstance(sigma_data, float):
            sigma_data = [sigma_data, sigma_data, sigma_data, sigma_data]
        if len(sigma_data) != 4:
            raise ValueError('sigma_data must be either a float or a list of 4 floats.')
        # self.sigma_data = sigma_data   # B,
        self.sigma_data = torch.tensor(sigma_data, device=device)   # 4,

    def get_scalings(self, sigma):   # B,
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)   # B, 4
        sigma = K.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)  # B, 4
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        # c_skip, c_out, c_in = [K.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]   # B,1,1
        c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma)]   # B,1,4
        noised_input = input + noise * K.utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        # c_skip, c_out, c_in = [K.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma)]   # B,1,4
        return self.inner_model(input * c_in, sigma, **kwargs) * c_out + input * c_skip






class PointwiseNet_kDiffusion(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        time_dim = 64
        fourier_scale = 16   # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling

        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+time_dim),
            ConcatSquashLinear(128, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 512, context_dim+time_dim),
            ConcatSquashLinear(512, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 128, context_dim+time_dim),
            ConcatSquashLinear(128, point_dim, context_dim+time_dim)
        ])

        self.timestep_embed = nn.Sequential(
            K.layers.FourierFeatures(1, time_dim, std=fourier_scale),   # 1D Fourier features --> with register_buffer, so weights are not trained
            nn.Linear(time_dim, time_dim), # this is a trainable layer
        )

    def forward(self, x, sigma, context=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, F). 
        """
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.act(self.timestep_embed(c_noise))  # (B, 1, T)
        
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+T)   # TODO: might want to add additional linear embedding net for context or only cond_feats

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out



class PointwiseNet_kDiffusion_Dropout(Module):

    def __init__(self, point_dim, context_dim, residual, dropout_rate=0.0):
        super().__init__()
        time_dim = 64
        fourier_scale = 16   # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling

        self.act = F.leaky_relu
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+time_dim),
            ConcatSquashLinear(128, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 512, context_dim+time_dim),
            ConcatSquashLinear(512, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 128, context_dim+time_dim),
            ConcatSquashLinear(128, point_dim, context_dim+time_dim)
        ])

        self.timestep_embed = nn.Sequential(
            K.layers.FourierFeatures(1, time_dim, std=fourier_scale),   # 1D Fourier features --> with register_buffer, so weights are not trained
            nn.Linear(time_dim, time_dim), # this is a trainable layer
        )

    def forward(self, x, sigma, context=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, F). 
        """
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.act(self.timestep_embed(c_noise))  # (B, 1, T)
        
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+T)   # TODO: might want to add additional linear embedding net for context or only cond_feats

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:    # no activation and dropout after last layer
                out = self.act(out)
                out = self.dropout(out)

        if self.residual:
            return x + out
        else:
            return out

