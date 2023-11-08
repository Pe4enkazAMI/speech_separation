import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import pyloudnorm as pyln

from hw_ss.base import BaseTrainer
from hw_ss.base.base_text_encoder import BaseTextEncoder
from hw_ss.logger.utils import plot_spectrogram_to_buf
from hw_ss.metric.utils import calc_sisdr
from hw_ss.utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50
        self.accum_steps = config["trainer"].get("accum_steps", 1)
        
        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )
        self.meter = pyln.Meter(16000)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["audio_mix", "audio_ref", "audio_target", "speaker_id"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _get_lr(self, optim):
        for param_group in optim.param_groups:
            return param_group["lr"]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                    index=batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "get_last_lr"):
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                else:
                    self.writer.add_scalar(
                        "learning rate", self._get_lr(self.optimizer)
                    )
                self._log_predictions(**batch, epoch=epoch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, index=None):
        batch = self.move_batch_to_device(batch, self.device)
        # if is_train:
        #     self.optimizer.zero_grad()
        if is_train:
            self.optimizer.zero_grad()
        s1, s2, s3, logits = self.model(batch["audio_mix"], batch["audio_ref"], batch["audio_ref_len"])
        batch["source_1"] = s1
        batch["source_2"] = s2
        batch["source_3"] = s3
        batch["logits"] = logits
        
        batch["loss"] = self.criterion(batch, is_train)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None and index == self.len_epoch - 1:
                self.lr_scheduler.step(batch["loss"].item())
            

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                    index=batch_idx
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, epoch=epoch)

        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            source_1,
            source_2,
            source_3,
            audio_mix,
            audio_ref,
            audio_target,
            speaker_id,
            audio_path_mix,
            examples_to_log=10,
            epoch=None,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        
        tuples = list(zip(source_1,
                           source_2,
                             source_3,
                               audio_mix,
                                 audio_ref,
                                   audio_target, audio_path_mix, speaker_id))
        
        
            
        shuffle(tuples)
        rows = {}
        
        for source_1, source_2, source_3, audio_mix, audio_ref, audio_target, audio_path_mix, speaker_id in tuples[:examples_to_log]:
            sisdr = calc_sisdr(source_1, audio_target)
            print(source_1.shape)
            louds1 = self.meter.integrated_loudness(source_1.squeeze(0).cpu().detach().numpy())
            source = pyln.normalize.loudness(source_1.squeeze(0).cpu().detach().numpy(), louds1, -29)
            rows[Path(audio_path_mix).name] = {
                "mix_audio": self.writer.wandb.Audio(audio_mix.squeeze(0).cpu().detach().numpy(), sample_rate=16000),
                "extracted_audio": self.writer.wandb.Audio(source, sample_rate=16000),
                "orig_audio": self.writer.wandb.Audio(audio_target.squeeze(0).cpu().detach().numpy(), sample_rate=16000),
                "reference_audio": self.writer.wandb.Audio(audio_ref.squeeze(0).cpu().detach().numpy(), sample_rate=16000),
                "SI-SDR": sisdr,
                "speaker id": speaker_id.item(),
                "epoch": epoch
            }
    
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
    
    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
