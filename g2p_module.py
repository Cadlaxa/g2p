import re
import hydra
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from k2 import rnnt_loss
from muon import SingleDeviceMuonWithAuxAdam
import editdistance
from dataset import PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, special_symbols


def _decode_tokens(token_ids: list[int], vocab: list[str]) -> list[str]:
    out = []
    for idx in token_ids:
        if idx == EOS_IDX:
            break
        if idx >= 4:
            out.append(vocab[idx])
    return out


class G2pModule(pl.LightningModule):
    def __init__(
        self,
        model: any,
        muon_lr: float,
        adam_lr: float,
        weight_decay: float,
        lr_scheduler: any,
        ilm_loss_weight: float,
        ilm_scheduled_sampling_ratio: float,
        grad_clip: float,
        loss_device: str,
        graphemes: list[str],
        phonemes: list[str],
        print_debug_preview: bool,
        logs_stressless_error_rates: bool,
        train_eval_interval: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.graphemes = graphemes
        self.phonemes = phonemes
        self.model = hydra.utils.instantiate(model)
        self.muon_lr = muon_lr
        self.adam_lr = adam_lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.ilm_loss_weight = ilm_loss_weight
        self.ilm_ss_ratio = ilm_scheduled_sampling_ratio
        self.grad_clip = grad_clip
        self.loss_device = torch.device(loss_device)
        self.print_debug_preview = print_debug_preview
        self.logs_stressless_error_rates = logs_stressless_error_rates
        self.train_eval_interval = train_eval_interval

        self.validation_step_outputs = []
        self.score_min = 1.0

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def configure_optimizers(self):
        hidden = []
        nonhidden = []
        for name, p in self.model.named_parameters():
            is_hidden = p.ndim >= 2 and not (
                hasattr(p, "is_muon_excluded") and p.is_muon_excluded
            )
            if is_hidden:
                hidden.append(p)
            else:
                nonhidden.append(p)

        param_groups = [
            dict(
                params=hidden,
                use_muon=True,
                lr=self.muon_lr,
                weight_decay=self.weight_decay,
            ),
            dict(
                params=nonhidden,
                use_muon=False,
                lr=self.adam_lr,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        lr_scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer=optimizer)
        result = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            result["lr_scheduler"]["monitor"] = "train/sampled_per"
            self.train_eval_interval = 1
        return result

    def training_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        B, T = src.shape

        tgt_in = torch.concat(
            [torch.full([B, 1], BOS_IDX, device=self.device, dtype=tgt.dtype), tgt],
            dim=1,
        )

        ilm_loss = torch.tensor(0.0, device=self.device)
        ss_prob = min(self.current_epoch / 50.0, 1.0) * self.ilm_ss_ratio
        use_ilm = self.ilm_loss_weight > 0 or ss_prob > 0

        dec_state = None
        if use_ilm:
            _, U = tgt_in.shape
            dec_state = self.model.decoder(tgt_in)
            ilm_enc = torch.zeros(
                (B, 1, self.model.encoder.d_model),
                device=self.device,
                dtype=dec_state.dtype,
            )
            ilm_logits = self.model.joint(ilm_enc, dec_state)
            ilm_logits = ilm_logits.squeeze(1)[:, :-1, :]
            ilm_logits[:, :, 0:EOS_IDX] = -1e10

            ilm_mask = torch.arange(U - 1, device=self.device).unsqueeze(0)
            ilm_mask = ilm_mask < tgt_lengths.unsqueeze(1)

            if self.ilm_loss_weight > 0:
                ilm_loss = F.cross_entropy(
                    ilm_logits.float().reshape(-1, ilm_logits.size(-1)),
                    tgt.to(torch.long).view(-1),
                    reduction="none",
                )
                ilm_loss = (
                    ilm_loss * ilm_mask.view(-1).float()
                ).sum() / ilm_mask.sum().float()

            if ss_prob > 0:
                with torch.no_grad():
                    ilm_logits[:, :, 0 : EOS_IDX + 1] = -1e10
                    ss_tgt = torch.argmax(ilm_logits, dim=-1).int()
                    ss_tgt = ss_tgt.masked_fill(~ilm_mask, PAD_IDX)
                    ss_tgt_in = torch.concat(
                        [
                            torch.full(
                                [B, 1], BOS_IDX, device=self.device, dtype=tgt.dtype
                            ),
                            ss_tgt,
                        ],
                        dim=1,
                    )
                    select_ss = torch.rand(B, U, device=self.device) < ss_prob
                    tgt_in = torch.where(select_ss, ss_tgt_in, tgt_in).detach()
                dec_state = None

        enc_state = self.model.encoder(src)
        if dec_state is None:
            dec_state = self.model.decoder(tgt_in)
        logits = self.model.joint(enc_state, dec_state)

        boundary = torch.zeros((B, 4), dtype=torch.int64, device=self.device)
        boundary[:, 2] = tgt_lengths
        boundary[:, 3] = src_lengths
        boundary = boundary.to(self.loss_device)
        rnnt_loss_val = rnnt_loss(
            logits=logits.float().to(self.loss_device),
            symbols=tgt.to(self.loss_device),
            termination_symbol=BOS_IDX,
            boundary=boundary,
        )

        total_loss = rnnt_loss_val + self.ilm_loss_weight * ilm_loss

        self.log(
            "train/loss", rnnt_loss_val, on_step=False, on_epoch=True, prog_bar=True
        )
        if self.ilm_loss_weight > 0:
            self.log("train/ilm_loss", ilm_loss, on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.train_eval_interval != 0:
            return

        train_sampled_loader = self.trainer.datamodule.train_sampled_dataloader()

        word_errors = 0
        word_count = 0
        phoneme_errors = 0
        phoneme_count = 0
        word_errors_stressless = 0
        phoneme_errors_stressless = 0
        phoneme_count_stressless = 0

        self.model.eval()
        with torch.no_grad():
            for batch in train_sampled_loader:
                decode_results = self.test_decode(batch)
                word_errors += decode_results["word_errors"]
                word_count += decode_results["word_count"]
                phoneme_errors += decode_results["phoneme_errors"]
                phoneme_count += decode_results["phoneme_count"]
                if self.logs_stressless_error_rates:
                    word_errors_stressless += decode_results.get(
                        "word_errors_stressless", 0
                    )
                    phoneme_errors_stressless += decode_results.get(
                        "phoneme_errors_stressless", 0
                    )
                    phoneme_count_stressless += decode_results.get(
                        "phoneme_count_stressless", 0
                    )

        self.model.train()

        sampled_wer = word_errors / word_count
        sampled_per = phoneme_errors / phoneme_count
        self.log("train/sampled_wer", sampled_wer * 100, prog_bar=True)
        self.log("train/sampled_per", sampled_per * 100, prog_bar=True)

        if self.logs_stressless_error_rates:
            sampled_wer_stressless = word_errors_stressless / word_count
            sampled_per_stressless = (
                phoneme_errors_stressless / phoneme_count_stressless
                if phoneme_count_stressless > 0
                else 0
            )
            self.log(
                "train/sampled_wer_stressless",
                sampled_wer_stressless * 100,
                prog_bar=False,
            )
            self.log(
                "train/sampled_per_stressless",
                sampled_per_stressless * 100,
                prog_bar=False,
            )

    def validation_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        B = src.shape[0]

        tgt_in = torch.concat(
            [torch.full([B, 1], BOS_IDX, device=self.device, dtype=tgt.dtype), tgt],
            dim=1,
        )
        logits = self.model(src, tgt_in)

        boundary = torch.zeros((B, 4), dtype=torch.int64, device=self.device)
        boundary[:, 2] = tgt_lengths
        boundary[:, 3] = src_lengths
        boundary = boundary.to(self.loss_device)
        loss = rnnt_loss(
            logits=logits.float().to(self.loss_device),
            symbols=tgt.to(self.loss_device),
            termination_symbol=BOS_IDX,
            boundary=boundary,
        )

        decode_results = self.test_decode(batch)
        self.validation_step_outputs.append({"val_loss": loss, **decode_results})
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        total_word_errors = sum([x["word_errors"] for x in outputs])
        total_word_count = sum([x["word_count"] for x in outputs])
        total_phoneme_errors = sum([x["phoneme_errors"] for x in outputs])
        total_phoneme_count = sum([x["phoneme_count"] for x in outputs])

        wer = total_word_errors / total_word_count if total_word_count > 0 else 0
        per = (
            total_phoneme_errors / total_phoneme_count if total_phoneme_count > 0 else 0
        )
        score = 0.5 * wer + 0.5 * per
        if score < self.score_min:
            self.score_min = score

        self.log("val/loss", avg_loss, prog_bar=True)
        self.log("val/wer", wer * 100, prog_bar=True)
        self.log("val/per", per * 100, prog_bar=True)
        self.log("val/score", score * 100, prog_bar=True)
        self.log("val/score_min", self.score_min * 100, prog_bar=True)
        self.log("hp_metric", self.score_min * 100)

        if self.logs_stressless_error_rates:
            total_word_errors_stressless = sum(
                [x.get("word_errors_stressless", 0) for x in outputs]
            )
            total_phoneme_errors_stressless = sum(
                [x.get("phoneme_errors_stressless", 0) for x in outputs]
            )
            total_phoneme_count_stressless = sum(
                [x.get("phoneme_count_stressless", 0) for x in outputs]
            )
            wer_stressless = (
                total_word_errors_stressless / total_word_count
                if total_word_count > 0
                else 0
            )
            per_stressless = (
                total_phoneme_errors_stressless / total_phoneme_count_stressless
                if total_phoneme_count_stressless > 0
                else 0
            )
            self.log("val/wer_stressless", wer_stressless * 100, prog_bar=False)
            self.log("val/per_stressless", per_stressless * 100, prog_bar=False)

        if self.print_debug_preview and len(outputs) > 0:
            first_results = outputs[0].get("previews", [])
            self.print("====== Validation Preview ======")
            for word, gold, pred in first_results:
                self.print(f"{''.join(word)}: [{'-'.join(gold)}] [{'-'.join(pred)}]")
            self.print("================================")

        self.validation_step_outputs.clear()

    def test_decode(self, batch):
        src, tgt, src_lengths, tgt_lengths = batch
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_lengths = src_lengths.to(self.device)

        tgt_out = self.model.greedy_decode(src, src_lengths, return_early=True)

        src_cpu = src.cpu().tolist()
        tgt_cpu = tgt.cpu().tolist()
        pred_cpu = tgt_out[:, 1:].cpu().tolist()

        word_errors = 0
        phoneme_errors = 0
        phoneme_count = 0
        word_errors_stressless = 0
        phoneme_errors_stressless = 0
        phoneme_count_stressless = 0

        word_count = len(src_cpu)
        previews = []
        for i in range(word_count):
            word = _decode_tokens(src_cpu[i], self.graphemes)
            gold = _decode_tokens(tgt_cpu[i], self.phonemes)
            pred = _decode_tokens(pred_cpu[i], self.phonemes)
            dis = editdistance.distance(gold, pred)
            phoneme_count += len(gold)
            if dis > 0:
                word_errors += 1
                phoneme_errors += dis
            if i < 5:
                previews.append((word, gold, pred))

            if self.logs_stressless_error_rates:
                gold_stressless = [re.sub(r"\d+$", "", p) for p in gold]
                pred_stressless = [re.sub(r"\d+$", "", p) for p in pred]
                dis_stressless = editdistance.distance(gold_stressless, pred_stressless)
                phoneme_count_stressless += len(gold_stressless)
                phoneme_errors_stressless += dis_stressless
                if dis_stressless > 0:
                    word_errors_stressless += 1

        result = {
            "word_errors": word_errors,
            "word_count": word_count,
            "phoneme_errors": phoneme_errors,
            "phoneme_count": phoneme_count,
            "previews": previews,
        }

        if self.logs_stressless_error_rates:
            result["word_errors_stressless"] = word_errors_stressless
            result["phoneme_errors_stressless"] = phoneme_errors_stressless
            result["phoneme_count_stressless"] = phoneme_count_stressless

        return result