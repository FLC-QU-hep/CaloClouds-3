# in the public repo this is CaloClouds_2.py
import torch
from torch.nn import Module, ModuleList, functional

from .common import ConcatSquashLinear, KLDloss, reparameterize_gaussian
from .encoders.epic_encoder_cond import EPiC_encoder_cond

# This import is a bit fragile, it only works if the directory above is on the python path
# That happens natrually if the directory above is where you started python
# but we could fix it by either expicitly adding the directory above to the path
# or by making a deeper directory structure so thi could be a relative import
from utils.misc import get_flow_model, mean_flat

import k_diffusion


class epicVAE_nFlow_kDiffusion(Module):

    def __init__(self, args, distillation = False):
        super().__init__()
        self.args = args
        self.distillation = distillation
        if args.latent_dim > 0:
            self.encoder = EPiC_encoder_cond(args.latent_dim, input_dim=args.features, cond_features=args.cond_features)
            self.flow = get_flow_model(args)

        if args.dropout_rate == 0.0:
            net = PointwiseNet_kDiffusion(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual)
        else:
            if args.dropout_mode == 'all':
                net = PointwiseNet_kDiffusion_Dropout(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual, dropout_rate=args.dropout_rate)
            elif args.dropout_mode == 'mid':
                net = PointwiseNet_kDiffusion_Dropout_mid(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual, dropout_rate=args.dropout_rate)
            else: 
                raise NotImplementedError('dropout mode not implemented')
        if not distillation:
            self.diffusion = Denoiser(net, sigma_data = args.model['sigma_data'], device=args.device, diffusion_loss=args.diffusion_loss)
        else:
            self.diffusion = Denoiser(net, sigma_data = args.model['sigma_data'], device=args.device, distillation=True, sigma_min=args.sigma_min)
        # self.diffusion = k_diffusion.config.make_denoiser_wrapper(args.__dict__)(net)
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

        if self.args.latent_dim > 0:
            z_mu, z_sigma = self.encoder(x, cond_feats)
            z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, functional)
            
            # VAE-like loss for encoder
            loss_kld = self.kld(mu=z_mu, logvar=z_sigma)
            loss_kld_clamped = torch.clamp(loss_kld, min=kld_min)

            # P(z), Prior probability, parameterized by the flow: z -> w.
            nll = - self.flow.log_prob(z.detach().clone(), cond_feats)    # detach from computational graph if optimizing encoder+diffuison seperate from flow
            loss_prior = nll.mean()

            # Diffusion MSE loss: Negative ELBO of P(X|z)
            z = torch.cat([z, cond_feats], -1)
        else:
            z = cond_feats
        #loss_diffusion = self.diffusion.get_loss(x, z)    # diffusion loss
        loss_diffusion = self.diffusion.loss(x, noise, sigma, context=z).mean()    # diffusion loss
        # print('max values: ', loss_diffusion.max().item(), loss_diffusion.min().item())
        # print('shape of loss_diffusion: ', loss_diffusion.shape)

        # Total loss
        if self.args.latent_dim > 0:
            loss = kl_weight*(loss_kld_clamped) + loss_diffusion
        else:
            loss_prior = None
            loss = loss_diffusion

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
        if self.args.latent_dim > 0:
            z = self.flow.sample(context=cond_feats, num_samples=1).view(batch_size, -1)  # B,functional
            z = torch.cat([z, cond_feats], -1)   # B, functional+C
        else:
            z = cond_feats  # B, C

        x_T = torch.randn([z.size(0), num_points, config.features], device=z.device) * config.sigma_max

        if not self.distillation:

            sigmas = k_diffusion.sampling.get_sigmas_karras(config.num_steps, config.sigma_min, config.sigma_max, rho=config.rho, device=z.device)

            sampler_kw_args = {'extra_args': {'context': z}, 'disable': True}
            if config.sampler == 'euler':
                sampler_class = k_diffusion.sampling.sample_euler
            elif config.sampler == 'heun':
                sampler_class = k_diffusion.sampling.sample_heun
                sampler_kw_args['s_churn'] = config.s_churn
                sampler_kw_args['s_noise'] = config.s_noise
            elif config.sampler == 'dpmpp_2m':
                sampler_class = k_diffusion.sampling.sample_dpmpp_2m
            elif config.sampler == 'dpmpp_2s_ancestral':
                sampler_class = k_diffusion.sampling.sample_dpmpp_2s_ancestral
            elif config.sampler == 'sample_euler_ancestral':
                sampler_class = k_diffusion.sampling.sample_euler_ancestral
            elif config.sampler == 'sample_lms':
                sampler_class = k_diffusion.sampling.sample_lms
            elif config.sampler == 'sample_dpmpp_2m_sde':
                sampler_class = k_diffusion.sampling.sample_dpmpp_2m_sde
            else:
                raise NotImplementedError(f'Sampler {config.sampler} not implemented')
            x_0 = sampler_class(self.diffusion, x_T, sigmas, **sampler_kw_args)
            
        else:  # one step for consistency model
            x_0 = self.diffusion.forward(x_T, config.sigma_max, context=z)

        return x_0
    
    # def inference(self, num_points, cfg, num_samples=256):
    #     with torch.no_grad():
    #         z = torch.randn([num_samples, cfg.latent_dim]).to(cfg.device)
    #         x = self.sample(z, num_points, flexibility=cfg.flexibility)
    #     return x

    # def sample_fromData(self, x, cond_feats, num_points, flexibility):
    #     z_mu, z_sigma = self.encoder(x, cond_feats)
    #     z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, functional)
    #     z = torch.cat([z, cond_feats], -1)   # B,functional+C
    #     samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
    #     return samples


    def get_cd_loss(self, x, cond_feats, model_teacher, model_target, config):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            cond_feats: conditional features, (B, C)
            model_teacher: teacher model as score function for ODE solver
            model_ema_target: target model
            config: config
        """

        # get latent code from encoder
        if self.args.latent_dim > 0:
            with torch.no_grad():
                z_mu, z_sigma = self.encoder(x, cond_feats)
                z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, functional)
                z = torch.cat([z, cond_feats], -1)   # B,functional+C
        else:
            z = cond_feats

        loss = self.diffusion.consistency_loss(x, model_teacher.diffusion, model_target.diffusion, config, context=z).mean()    # consistency loss

        return loss
    


# from: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py#L12
class Denoiser(torch.nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=0.5, device='cuda', distillation = False, sigma_min = 0.002, diffusion_loss='l2'):
        super().__init__()
        self.inner_model = inner_model
        if isinstance(sigma_data, float):
            sigma_data = [sigma_data, sigma_data, sigma_data, sigma_data]
        if len(sigma_data) != 4:
            raise ValueError('sigma_data must be either a float or a list of 4 floats.')
        # self.sigma_data = sigma_data   # B,
        self.sigma_data = torch.tensor(sigma_data, device=device)   # 4,
        self.distillation = distillation
        self.sigma_min = sigma_min
        self.diffusion_loss = diffusion_loss

    def get_scalings(self, sigma):   # B,
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)   # B, 4
        sigma = k_diffusion.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)  # B, 4
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        return c_skip, c_out, c_in
    
    def get_scalings_for_boundary_condition(self, sigma):   # B,   # for consistency model
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)   # B, 4
        sigma = k_diffusion.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + sigma_data**2
        )   # B, 4
        c_out = (
            (sigma - self.sigma_min)
            * sigma_data
            / (sigma**2 + sigma_data**2) ** 0.5
        )   # B, 4
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        # c_skip, c_out, c_in = [k_diffusion.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]   # B,1,1
        c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma)]   # B,1,4
        noised_input = input + noise * k_diffusion.utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        if self.diffusion_loss == 'l2':
            return (model_output - target).pow(2).flatten(1).mean(1)
        elif self.diffusion_loss == 'l1':
            return (model_output - target).abs().flatten(1).mean(1)
        else:
            raise ValueError('diffusion_loss must be either l1 or l2')

    def forward(self, input, sigma, **kwargs):   # same as "denoise" in KarrasDenoiser of CM code
        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = (
                torch.tensor([sigma] * input.shape[0], dtype=torch.float32)
                .to(input.device)
                .unsqueeze(1)
            )
        # c_skip, c_out, c_in = [k_diffusion.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        if not self.distillation:
            c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma)]   # B,1,4
        else:
            c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings_for_boundary_condition(sigma)]
        # CM code did an additional resacling of the time sigma for the time conditing
        return self.inner_model(input * c_in, sigma, **kwargs) * c_out + input * c_skip
    
    # inspired by https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L106
    def consistency_loss(self, input, teacher_model, target_model, config, **kwargs):

        noise = torch.randn_like(input)
        dims = input.ndim
        num_scales = config.num_steps


        def denoise_fn(x, t):   # t = sigma
            return self(x, t, **kwargs)
        
        @torch.no_grad()
        def target_denoise_fn(x, t):
            return target_model(x, t, **kwargs)

        @torch.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_model(x, t, **kwargs)


        @torch.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / k_diffusion.utils.append_dims(t, dims)
            samples = x + d * k_diffusion.utils.append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0     # but this would not be the correct Heun method any more? anyway, without teacher model it's using Euler
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / k_diffusion.utils.append_dims(next_t, dims)
            samples = x + (d + next_d) * k_diffusion.utils.append_dims((next_t - t) / 2, dims)

            return samples

        @torch.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / k_diffusion.utils.append_dims(t, dims)
            samples = x + d * k_diffusion.utils.append_dims(next_t - t, dims)

            return samples


        # Get random sigmas / EDM boundaries    same as k_diffusion.utils.get_sigmas_karras()   with t and t+1
        indices = torch.randint(
            0, num_scales - 1, (input.shape[0],), device=input.device
        )

        # in paper: t + 1
        t = config.sigma_max ** (1 / config.rho) + indices / (num_scales - 1) * (
            config.sigma_min ** (1 / config.rho) - config.sigma_max ** (1 / config.rho)
        )
        t = t**config.rho

        # in paper: t    --> so t > t2
        t2 = config.sigma_max ** (1 / config.rho) + (indices + 1) / (num_scales - 1) * (
            config.sigma_min ** (1 / config.rho) - config.sigma_max ** (1 / config.rho)
        )
        t2 = t2**config.rho

        x_t = input + noise * k_diffusion.utils.append_dims(t, dims)   # calculate x_t at time step t+1 from data

        dropout_state = torch.get_rng_state()   # get state of the random number generator
        distiller = denoise_fn(x_t, t)     # denoise x_t completely x_t (t + 1) --> x_0 = input

        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, input).detach()    # for consistency training, not used
        else:
            x_t2 = heun_solver(x_t, t, t2, input).detach()     # for consistency distllation, one solver step to get from t+1 to t

        torch.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)   # target model (ema, not trained) denoises data completely from time t to t=0 / x_0 / input
        distiller_target = distiller_target.detach()

        weights = 1.    # paper: uniform weights work well in their experiments

        # l2 loss / MSE loss
        diffs = (distiller - distiller_target) ** 2
        loss = mean_flat(diffs) * weights

        return loss







class PointwiseNet_kDiffusion(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        time_dim = 64
        fourier_scale = 16   # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling

        self.act = functional.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+time_dim),
            ConcatSquashLinear(128, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 512, context_dim+time_dim),
            ConcatSquashLinear(512, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 128, context_dim+time_dim),
            ConcatSquashLinear(128, point_dim, context_dim+time_dim)
        ])

        self.timestep_embed = torch.nn.Sequential(
            k_diffusion.layers.FourierFeatures(1, time_dim, std=fourier_scale),   # 1D Fourier features --> with register_buffer, so weights are not trained
            torch.nn.Linear(time_dim, time_dim), # this is a trainable layer
        )

    def forward(self, x, sigma, context=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, functional). 
        """
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, functional)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.act(self.timestep_embed(c_noise))  # (B, 1, T)
        
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, functional+T)   # TODO: might want to add additional linear embedding net for context or only cond_feats

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

        self.act = functional.leaky_relu
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+time_dim),
            ConcatSquashLinear(128, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 512, context_dim+time_dim),
            ConcatSquashLinear(512, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 128, context_dim+time_dim),
            ConcatSquashLinear(128, point_dim, context_dim+time_dim)
        ])

        self.timestep_embed = torch.nn.Sequential(
            k_diffusion.layers.FourierFeatures(1, time_dim, std=fourier_scale),   # 1D Fourier features --> with register_buffer, so weights are not trained
            torch.nn.Linear(time_dim, time_dim), # this is a trainable layer
        )

    def forward(self, x, sigma, context=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, functional). 
        """
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, functional)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.act(self.timestep_embed(c_noise))  # (B, 1, T)
        
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, functional+T)   # TODO: might want to add additional linear embedding net for context or only cond_feats

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

class PointwiseNet_kDiffusion_Dropout_mid(Module):

    def __init__(self, point_dim, context_dim, residual, dropout_rate=0.0):
        super().__init__()
        time_dim = 64
        fourier_scale = 16   # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling

        self.act = functional.leaky_relu
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+time_dim),
            ConcatSquashLinear(128, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 512, context_dim+time_dim),
            ConcatSquashLinear(512, 256, context_dim+time_dim),
            ConcatSquashLinear(256, 128, context_dim+time_dim),
            ConcatSquashLinear(128, point_dim, context_dim+time_dim)
        ])

        self.timestep_embed = torch.nn.Sequential(
            k_diffusion.layers.FourierFeatures(1, time_dim, std=fourier_scale),   # 1D Fourier features --> with register_buffer, so weights are not trained
            torch.nn.Linear(time_dim, time_dim), # this is a trainable layer
        )

    def forward(self, x, sigma, context=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, functional). 
        """
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, functional)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.act(self.timestep_embed(c_noise))  # (B, 1, T)
        
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, functional+T)   # TODO: might want to add additional linear embedding net for context or only cond_feats

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:    # no activation and dropout after last layer
                out = self.act(out)
                if i == len(self.layers) - 3:
                    out = self.dropout(out)

        if self.residual:
            return x + out
        else:
            return out
