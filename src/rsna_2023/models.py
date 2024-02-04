import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lightning_utilities.core.rank_zero import rank_zero_only
from hydra.utils import instantiate
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from rsna_2023.videomaev2 import vit_small_patch16_224


def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def initialize_net(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class RNNLitModel(pl.LightningModule):
    def __init__(
            self,
            net,
            rnn,
            head,
            optimizer,
            scheduler_dict,
            scheduler,
            num_frames
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = instantiate(
            net
        )
        self.rnn = instantiate(
            rnn,
            input_size=self.net.num_features
        )
        self.head = instantiate(
            head
        )

        self.class_weights = torch.tensor([1, 2, 4]).float()
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=0.05
        )

        self.validation_step_labels = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = instantiate(
            self.hparams.optimizer,
            params=(
                add_weight_decay(self.trainer.model, self.hparams.optimizer['weight_decay'])
                if 'weight_decay' in self.hparams.optimizer
                else self.trainer.model.parameters()
            )
        )
        scheduler_dict = self.hparams.scheduler_dict
        scheduler_dict['scheduler'] = instantiate(
            self.hparams.scheduler, optimizer=optimizer
        )
        return [optimizer], [self.hparams.scheduler_dict]

    def predict_step(self, batch, batch_idx, dataloader_idx=0, split=None):
        inputs = batch['image']
        batch_size, input_channels, x_size, y_size, z_size = inputs.shape

        step = x_size // self.hparams.num_frames

        if split == 'train':
            start_z = torch.randint(1, step -1, (1,)).item()
        else:
            start_z = step // 2

        xy = torch.concat(
            [
                inputs[:, :, :, :, start_z-1::step],
                inputs[:, :, :, :, start_z::step],
                inputs[:, :, :, :, start_z+1::step]
            ],
            dim=1
        )  # [bs, ch, x, y, z]
        xy = xy.permute(0, 4, 1, 2, 3).contiguous()  # [bs, z, ch, x, y]
        batch_size, z_size, input_channels, x_size, y_size = xy.shape

        x = xy

        x = x.view(batch_size * z_size, input_channels, x_size, y_size)
        x = self.net(x)
        x = x.view(batch_size, z_size, -1)
        output, (h_n, c_n) = self.rnn(x)
        h_n = h_n.view(2 if self.rnn.bidirectional else 1, self.rnn.num_layers, batch_size, self.rnn.hidden_size)[:, -1]
        x = h_n.transpose(0, 1).contiguous().view(batch_size, -1)
        x = torch.cat([
            x,
            batch['aortic_hu'].float().view(-1, 1)
        ], dim=1)
        outputs = self.head(x)
        return outputs

    def calculate_loss(self, batch, outputs, split):
        labels = batch['labels']
        loss = self.criterion(
            outputs, labels.argmax(dim=1)
        )

        self.log(f'{split}_loss', loss, prog_bar=False, sync_dist=True, batch_size=labels.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self.predict_step(batch, batch_idx, split='train')
        loss = self.calculate_loss(batch, outputs, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.predict_step(batch, batch_idx)
        loss = self.calculate_loss(batch, outputs, 'val')
        self.validation_step_labels.append(batch['labels'].detach().cpu().float())
        self.validation_step_outputs.append(outputs.detach().cpu().float())
        return loss

    def on_validation_epoch_end(self):
        labels = torch.cat(self.validation_step_labels, dim=0)
        outputs = torch.cat(self.validation_step_outputs, dim=0)
        labels = self.all_gather(labels).reshape(-1, *labels.shape[1:]).cpu()
        outputs = self.all_gather(outputs).reshape(-1, *outputs.shape[1:]).cpu()
        outputs_proba = F.softmax(outputs, dim=1)

        labels_argmax = labels.argmax(dim=1)
        sample_weights = torch.take(self.class_weights, labels_argmax)
        val_log_loss = log_loss(
            y_true=labels_argmax,
            y_pred=outputs_proba,
            sample_weight=sample_weights,
            labels=[0, 1, 2]
        )
        self.log('val_log_loss', val_log_loss, prog_bar=False)

        if len(labels_argmax.unique()) > 2:
            val_roc_auc_score = roc_auc_score(
                labels_argmax,
                outputs_proba,
                labels=[0, 1, 2],
                multi_class='ovr'
            )
        else:
            val_roc_auc_score = 0
        self.log('val_roc_auc', val_roc_auc_score, prog_bar=False)

        self.validation_step_labels.clear()
        self.validation_step_outputs.clear()


class VideoMAEv2Wrapper(nn.Module):
    def __init__(self, pth=None, num_labels=3):
        super().__init__()
        self.module = vit_small_patch16_224(num_classes=710)
        if pth is not None:
            state_dict = torch.load(pth, map_location='cpu')['module']
            self.module.load_state_dict(state_dict)
        self.module.head = nn.Linear(self.module.head.in_features, num_labels)
        initialize_net(self.module.head)
        self.num_frames = 16

    def forward(self, x):
        return self.module(x)


class Classification3DLitModel(pl.LightningModule):
    def __init__(
            self,
            net,
            optimizer,
            scheduler_dict,
            scheduler
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = instantiate(
            net
        )

        self.class_weights = torch.tensor([1, 2, 4]).float()
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=0.05
        )

        self.validation_step_labels = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = instantiate(
            self.hparams.optimizer,
            params=(
                add_weight_decay(self.trainer.model, self.hparams.optimizer['weight_decay'])
                if 'weight_decay' in self.hparams.optimizer
                else self.trainer.model.parameters()
            )
        )
        scheduler_dict = self.hparams.scheduler_dict
        scheduler_dict['scheduler'] = instantiate(
            self.hparams.scheduler, optimizer=optimizer
        )
        return [optimizer], [self.hparams.scheduler_dict]

    def predict_step(self, batch, batch_idx, dataloader_idx=0, split=None):
        inputs = batch['image']
        batch_size, input_channels, x_size, y_size, z_size = inputs.shape

        step = x_size // self.net.num_frames

        if split == 'train':
            start_z = torch.randint(1, step-1, (1,)).item()
        else:
            start_z = step // 2

        xy = torch.concat(
            [
                inputs[:, :, :, :, start_z-1::step],
                inputs[:, :, :, :, start_z::step],
                inputs[:, :, :, :, start_z+1::step]
            ],
            dim=1
        )
        xy = xy.permute(0, 1, 4, 2, 3).contiguous()  # [bs, ch, z, x, y]
        pixel_values = xy
        outputs = self.net(pixel_values)
        return outputs


    def calculate_loss(self, batch, outputs, split):
        labels = batch['labels']
        loss = self.criterion(
            outputs, labels.argmax(dim=1)
        )
        self.log(f'{split}_loss', loss, prog_bar=False, sync_dist=True, batch_size=labels.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self.predict_step(batch, batch_idx, split='train')
        loss = self.calculate_loss(batch, outputs, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.predict_step(batch, batch_idx)
        loss = self.calculate_loss(batch, outputs, 'val')
        self.validation_step_labels.append(batch['labels'].detach().cpu().float())
        self.validation_step_outputs.append(outputs.detach().cpu().float())
        return loss

    def on_validation_epoch_end(self):
        labels = torch.cat(self.validation_step_labels, dim=0)
        outputs = torch.cat(self.validation_step_outputs, dim=0)
        labels = self.all_gather(labels).reshape(-1, *labels.shape[1:]).cpu()
        outputs = self.all_gather(outputs).reshape(-1, *outputs.shape[1:]).cpu()
        outputs_proba = F.softmax(outputs, dim=1)
        labels_argmax = labels.argmax(dim=1)
        sample_weights = torch.take(self.class_weights, labels_argmax)
        val_log_loss = log_loss(
            y_true=labels_argmax,
            y_pred=outputs_proba,
            sample_weight=sample_weights,
            labels=[0, 1, 2]
        )
        self.log('val_log_loss', val_log_loss, prog_bar=False)

        if len(labels_argmax.unique()) > 2:
            val_roc_auc_score = roc_auc_score(
                labels_argmax,
                outputs_proba,
                labels=[0, 1, 2],
                multi_class='ovr'
            )
        else:
            val_roc_auc_score = 0
        self.log('val_roc_auc', val_roc_auc_score, prog_bar=False)

        self.validation_step_labels.clear()
        self.validation_step_outputs.clear()


def patch_first_conv2d(model):
    weight = model.conv1[0].weight.sum(dim=1, keepdim=True)
    bias = model.conv1[0].bias
    model.conv1[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.conv1[0].weight = nn.Parameter(weight)
    model.conv1[0].bias = bias
    return model


class SeqRNNLitModel(pl.LightningModule):
    def __init__(
            self,
            net,
            rnn,
            head,
            optimizer,
            scheduler_dict,
            scheduler,
            num_frames
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = instantiate(
            net
        )
        patch_first_conv2d(self.net)
        self.rnn = instantiate(
            rnn,
            input_size=self.net.num_features
        )
        self.head = instantiate(
            head
        )

        self.bowel_weights = torch.tensor([1, 2]).float()
        self.extravasation_weights = torch.tensor([1, 6]).float()
        self.class_weights = torch.tensor([2, 6]).float()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.class_weights
        )

        self.validation_step_labels = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = instantiate(
            self.hparams.optimizer,
            params=(
                add_weight_decay(self.trainer.model, self.hparams.optimizer['weight_decay'])
                if 'weight_decay' in self.hparams.optimizer
                else self.trainer.model.parameters()
            )
        )
        scheduler_dict = self.hparams.scheduler_dict
        scheduler_dict['scheduler'] = instantiate(
            self.hparams.scheduler, optimizer=optimizer
        )
        return [optimizer], [self.hparams.scheduler_dict]

    def predict_step(self, batch, batch_idx, dataloader_idx=0, split=None):
        inputs = batch['image']
        batch_size, input_channels, x_size, y_size, z_size = inputs.shape

        step = x_size // self.hparams.num_frames

        if split == 'train':
            start_z = torch.randint(0, step, (1,)).item()
        else:
            start_z = step // 2

        xy = inputs[:, :, :, :, start_z::step]
        xy = xy.permute(0, 4, 1, 2, 3).contiguous()  # [bs, z, ch, x, y]
        batch_size, z_size, input_channels, x_size, y_size = xy.shape

        x = xy
        x = x.view(batch_size * z_size, input_channels, x_size, y_size)
        x = self.net(x)
        x = x.view(batch_size, z_size, -1)
        output, (h_n, c_n) = self.rnn(x)
        x = output.reshape(batch_size * z_size, (2 if self.rnn.bidirectional else 1) * self.rnn.hidden_size)
        x = torch.cat(
            [
                x,
                batch['aortic_hu'].float().view(-1, 1).repeat(z_size, 1)
            ],
            dim=1
        )
        outputs = self.head(x)
        return outputs, start_z, step


    def calculate_loss(self, batch, outputs, start_z, step, split):
        labels = batch['labels']

        bowel_injury_seq = batch['bowel_injury']
        active_extravasation_seq = batch['active_extravasation']
        seq_labels = torch.cat(
            [
                bowel_injury_seq,
                active_extravasation_seq
            ],
            dim=1
        )
        seq_labels = seq_labels[:, :, start_z::step]
        batch_size, n_labels, z_size = seq_labels.shape
        seq_labels = seq_labels.permute(0, 2, 1).reshape(batch_size * z_size, n_labels)
        loss = self.criterion(
            outputs, seq_labels
        )
        self.log(f'{split}_loss', loss, prog_bar=False, sync_dist=True, batch_size=labels.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        outputs, start_z, step = self.predict_step(batch, batch_idx, split='train')
        loss = self.calculate_loss(batch, outputs, start_z, step, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, start_z, step = self.predict_step(batch, batch_idx)
        loss = self.calculate_loss(batch, outputs, start_z, step, 'val')

        batch_size = batch['labels'].shape[0]
        z_size = self.hparams.num_frames
        self.validation_step_labels.append(batch['labels'].detach().cpu().float())
        self.validation_step_outputs.append(
            outputs.reshape(
                batch_size, z_size, -1
            ).max(dim=1).values.detach().cpu().float()
        )
        return loss

    def on_validation_epoch_end(self):
        labels = torch.cat(self.validation_step_labels, dim=0)
        outputs = torch.cat(self.validation_step_outputs, dim=0)
        labels = self.all_gather(labels).reshape(-1, *labels.shape[1:]).cpu()
        outputs = self.all_gather(outputs).reshape(-1, *outputs.shape[1:]).cpu()
        outputs_proba = F.sigmoid(outputs)

        b_labels_argmax = labels[:, :2].argmax(dim=1)
        b_sample_weights = torch.take(self.bowel_weights, b_labels_argmax)
        b_val_log_loss = log_loss(
            y_true=b_labels_argmax,
            y_pred=outputs_proba[:, 0],
            sample_weight=b_sample_weights,
            labels=[0, 1]
        )
        self.log('val_bowel_log_loss', b_val_log_loss, prog_bar=False)

        e_labels_argmax = labels[:, 2:].argmax(dim=1)
        e_sample_weights = torch.take(self.class_weights, e_labels_argmax)
        e_val_log_loss = log_loss(
            y_true=e_labels_argmax,
            y_pred=outputs_proba[:, 1],
            sample_weight=e_sample_weights,
            labels=[0, 1]
        )
        self.log('val_extravasation_log_loss', e_val_log_loss, prog_bar=False)

        self.log('val_log_loss', (b_val_log_loss + e_val_log_loss) / 2, prog_bar=False)

        if len(b_labels_argmax.unique()) > 1:
            b_val_roc_auc_score = roc_auc_score(
                b_labels_argmax,
                outputs_proba[:, 0],
                labels=[0, 1],
                multi_class='ovr'
            )
        else:
            b_val_roc_auc_score = 0
        self.log('val_bowel_roc_auc', b_val_roc_auc_score, prog_bar=False)

        if len(e_labels_argmax.unique()) > 1:
            e_val_roc_auc_score = roc_auc_score(
                e_labels_argmax,
                outputs_proba[:, 1],
                labels=[0, 1],
                multi_class='ovr'
            )
        else:
            e_val_roc_auc_score = 0
        self.log('val_extravasation_roc_auc', e_val_roc_auc_score, prog_bar=False)


        self.validation_step_labels.clear()
        self.validation_step_outputs.clear()


# https://github.com/Lightning-Universe/lightning-bolts/blame/master/src/pl_bolts/optimizers/lr_scheduler.py
import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler; please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
