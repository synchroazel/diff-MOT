import enum
from functools import partial
from typing import Any

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kornia.geometry.transform import Rotate as krot
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm
from transformers.optimization import Adafactor

matplotlib.use("agg")


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def rotate_images(patches, rot_vector):
    angle_vec = rot_vector  # x_noisy[:, -2:]
    angles = -torch.atan2(angle_vec[:, 1], angle_vec[:, 0]) / torch.pi * 180
    r = krot(angles, mode="nearest")
    rot2 = r(patches)
    return rot2


def cosine_beta_schedule(timesteps, s=0.08):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape=None):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out[:, None]


@torch.jit.script
def greedy_cost_assignment(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    # Compute pairwise distances between positions
    dist = torch.norm(pos1[:, None] - pos2, dim=2)

    # Create a tensor to store the assignments
    assignments = torch.zeros(dist.size(0), 3, dtype=torch.int64)

    # Create a mask to keep track of assigned positions
    mask = torch.ones_like(dist, dtype=torch.bool)

    # Counter for keeping track of the number of assignments
    counter = 0

    # While there are still unassigned positions
    while mask.sum() > 0:
        # Find the minimum distance
        min_val, min_idx = dist[mask].min(dim=0)

        # Get the indices of the two dimensions
        idx = int(min_idx.item())
        ret = mask.nonzero()[idx, :]
        i = ret[0]
        j = ret[1]

        # Add the assignment to the tensor
        assignments[counter, 0] = i
        assignments[counter, 1] = j
        assignments[counter, 2] = min_val

        # Increase the counter
        counter += 1

        # Remove the assigned positions from the distance matrix and the mask
        mask[i, :] = 0
        mask[:, j] = 0

    return assignments[:counter]


class GNN_Diffusion(pl.LightningModule):
    def __init__(
            self,
            steps=600,
            inference_ratio=1,
            sampling="DDPM",
            learning_rate=1e-4,
            save_and_sample_every=1000,
            classifier_free_prob=0,
            classifier_free_w=0,
            noise_weight=0.0,
            rotation=False,
            model_mean_type: ModelMeanType = ModelMeanType.START_X,
            custom_gnn=None,
            mps_fallback=False,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_mean_type = model_mean_type
        self.learning_rate = learning_rate
        self.save_and_sample_every = save_and_sample_every
        self.classifier_free_prob = classifier_free_prob
        self.classifier_free_w = classifier_free_w
        self.noise_weight = noise_weight
        self.rotation = rotation

        if sampling == "DDPM":
            self.inference_ratio = 1
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddpm,
            )
            self.eta = 1

        elif sampling == "DDIM":
            self.inference_ratio = inference_ratio
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddim,
            )
            self.eta = 0

        # Define beta schedule
        betas = linear_beta_schedule(timesteps=steps)

        self.register_buffer("betas", betas)

        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)

        self.steps = steps

        self.model = custom_gnn

        self.save_hyperparameters()

    def __str__(self):
        return f"GNN_Diffusion_{self.steps}_" + str(self.model)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch) -> Any:
        return self.model(xy_pos, time, patch_rgb, edge_index, batch)

    def forward_with_feats(
            self,
            xy_pos: Tensor,
            time: Tensor,
            node_feats: Tensor,
            edge_feats: Tensor,
            edge_index: Tensor,
            batch,
    ) -> Any:
        return self.model.forward_with_feats(
            xy_pos, time, node_feats, edge_feats, edge_index, batch
        )

    # Forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
            self,
            x_start,
            t,
            noise=None,
            loss_type="l1",
            node_feats=None,
            edge_feats=None,
            edge_index=None,
            batch=None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start.float())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.steps == 1:  # Transformer case
            x_noisy = torch.zeros_like(x_noisy)

        prediction = self.forward_with_feats(
            x_noisy,
            t,
            node_feats,
            edge_feats,
            edge_index,
            batch=batch,
        )

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]

        if loss_type == "l1":
            loss = F.l1_loss(target, prediction)
        elif loss_type == "l2":
            loss = F.mse_loss(target, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(target, prediction)
        elif loss_type == "focal":
            loss = sigmoid_focal_loss(prediction, target.float(), reduction="mean")
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample_ddpm(self, x, t, t_index, node_feats, edge_feats, edge_index, patch_feats, batch):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        out = self.forward_with_feats(x, t, node_feats, edge_feats, edge_index, batch=batch)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * out / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = extract(
            self.alphas_cumprod, timestep
        )

        alpha_prod_t_prev = (
            extract(self.alphas_cumprod, prev_timestep)
            if (prev_timestep >= 0).all()
            else alpha_prod_t * 0 + 1
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
                1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    @torch.no_grad()
    def p_sample_ddim(
            self, x, t, t_index, node_feats, edge_feats, edge_index, patch_feats, batch
    ):

        prev_timestep = t - self.inference_ratio

        eta = self.eta
        alpha_prod = extract(self.alphas_cumprod, t, x.shape)

        if (prev_timestep >= 0).all():
            alpha_prod_prev = extract(self.alphas_cumprod, prev_timestep, x.shape)
        else:
            alpha_prod_prev = alpha_prod * 0 + 1

        beta = 1 - alpha_prod
        beta_prev = 1 - alpha_prod_prev

        if self.classifier_free_prob > 0.0:
            model_output_cond = self.forward_with_feats(
                x, t, node_feats, edge_feats, edge_index, batch=batch
            )

            model_output_uncond = self.forward_with_feats(
                x,
                t,
                node_feats,
                edge_feats,
                edge_index,
                batch=batch,
            )
            model_output = (
                                   1 + self.classifier_free_w
                           ) * model_output_cond - self.classifier_free_w * model_output_uncond
        else:
            model_output = self.forward_with_feats(
                x, t, node_feats, edge_feats, edge_index, batch=batch
            )

        # Estimate x_0

        x_0 = {
            ModelMeanType.EPSILON: (x - beta ** 0.5 * model_output) / alpha_prod ** 0.5,
            ModelMeanType.START_X: model_output,
        }[self.model_mean_type]
        eps = self._predict_eps_from_xstart(x, t, x_0)

        variance = self._get_variance(
            t, prev_timestep
        )

        std_eta = eta * variance ** 0.5

        pred_sample_direction = (1 - alpha_prod_prev - std_eta ** 2) ** (0.5) * eps

        prev_sample = alpha_prod_prev ** (0.5) * x_0 + pred_sample_direction

        if eta > 0:
            noise = torch.randn(model_output.shape, dtype=model_output.dtype).to(
                self.device
            )
            prev_sample = prev_sample + std_eta * noise
        return prev_sample

    @torch.no_grad()
    def p_sample_loop(self, shape, node_feats, edge_feats, edge_index, batch=None):
        device = self.device

        b = shape[0]

        # Start from pure noise
        img = torch.randn(shape, device=device) * self.noise_weight

        imgs = [img]

        for i in tqdm(
                list(reversed(range(0, self.steps, self.inference_ratio))),
                desc="[TQDM] Sampling loop time step", leave=False
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i,
                node_feats=node_feats,
                edge_feats=edge_feats,
                edge_index=edge_index,
                batch=batch,
            )

        imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(
            self, x, t, t_index, node_feats, edge_feats, edge_index, sampling_func, batch
    ):
        return sampling_func(x, t, t_index, node_feats, edge_feats, edge_index, sampling_func, batch)

    @torch.no_grad()
    def sample(
            self,
            image_size,
            batch_size=16,
            channels=3,
            cond=None,
            edge_index=None,
            batch=None,
    ):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            cond=cond,
            edge_index=edge_index,
            batch=batch,
        )

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters())
        return optimizer
