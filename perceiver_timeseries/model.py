# perceiver_timeseries/model.py
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from perceiver.model.core import PerceiverEncoder, PerceiverDecoder
from perceiver.model.core.position import FourierPositionEncoding

class MultivariatePerceiver(LightningModule):
    def __init__(self,
                 num_input_channels=7,
                 in_len=4096,
                 out_len=5000,
                 num_latents=256,
                 latent_channels=128,
                 num_layers=8,
                 learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1. 把 (T, C) 展平成序列长度 T * C_feat
        self.input_proj = nn.Linear(num_input_channels, latent_channels)

        # 2. Positional encoding (Fourier) 与官方实现保持一致
        self.pos_enc = FourierPositionEncoding(
             input_shape=(self.hparams.in_len,),   # 1-D sequence length
             num_frequency_bands=64,               # ✅ new kwarg
             concat_pos=False,                     # drops “original coords” stem
            sine_only=False,                      # keep sin+cos
        )


        self.encoder = PerceiverEncoder(
            num_latents=num_latents,
            num_latent_channels=latent_channels,
            num_cross_attention_layers=1,
            num_self_attention_blocks=num_layers,
            num_self_attention_layers_per_block=1)

        # 3. 生成 out_len × channels 的查询向量
        self.query = nn.Parameter(torch.randn(out_len, latent_channels))

        self.decoder = PerceiverDecoder(
            output_query_channels=latent_channels,
            output_num_channels=num_input_channels)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: [B, in_len, C]
        x = self.input_proj(x) + self.pos_enc(x)            # [B, in_len, latent_channels]
        latents = self.encoder(x)                           # [B, num_latents, latent_channels]
        query = self.query.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, out_len, latent_channels]
        preds = self.decoder(latents, query)                # [B, out_len, C]
        return preds

    def training_step(self, batch, _):
        y_hat = self(batch["inputs"])
        loss = self.loss_fn(y_hat, batch["targets"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        y_hat = self(batch["inputs"])
        loss = self.loss_fn(y_hat, batch["targets"])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
