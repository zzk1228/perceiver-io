import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from perceiver.model.core import (
    InputAdapter,
    OutputAdapter,
    PerceiverEncoder,
    PerceiverDecoder,
    TrainableQueryProvider,
)
from perceiver.model.core.position import FourierPositionEncoding


class TimeSeriesInputAdapter(InputAdapter):
    """Projects multivariate sequences and adds Fourier position encodings."""

    def __init__(self, num_input_channels: int, seq_len: int, latent_channels: int, num_frequency_bands: int = 64):
        super().__init__(num_input_channels=latent_channels)
        self.linear = nn.Linear(num_input_channels, latent_channels)
        self.pos_proj = nn.Linear(1 + 2*num_frequency_bands, latent_channels, bias=False)
        self.position_encoding = FourierPositionEncoding(
            input_shape=(seq_len,),

            num_frequency_bands=num_frequency_bands,
  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.linear(x)                     # (B,L,256)
        pos = self.position_encoding(b)        # (B,L,129)
        pos = self.pos_proj(pos)               # (B,L,256)
        return x + pos

class TimeSeriesOutputAdapter(OutputAdapter):
    """Maps decoder outputs to target channels."""

    def __init__(self, num_output_query_channels: int, num_output_channels: int):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, num_output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# perceiver_timeseries/model.py

class MultivariatePerceiver(LightningModule):
    def __init__(
        self,
        num_input_channels: int = 7,
        in_len: int = 5000,
        out_len: int = 5000,
        num_latents: int = 256,
        latent_channels: int = 256,
        num_layers: int = 8,
        learning_rate: float = 1e-4,
        num_cross_attention_heads: int = 1,  # <-- ADD THIS LINE
        num_self_attention_heads: int = 1,   # <-- ADD THIS LINE
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        input_adapter = TimeSeriesInputAdapter(
            num_input_channels=num_input_channels,
            seq_len=in_len,
            latent_channels=latent_channels,
        )

        self.encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=num_latents,
            num_latent_channels=latent_channels,
            num_cross_attention_layers=1,
            num_cross_attention_heads=num_cross_attention_heads,  # <-- ADD THIS LINE
            num_self_attention_blocks=num_layers,
            num_self_attention_layers_per_block=1,
            num_self_attention_heads=num_self_attention_heads,    # <-- ADD THIS LINE
        )

        query_provider = TrainableQueryProvider(
            num_queries=out_len,
            num_query_channels=latent_channels,
        )
        output_adapter = TimeSeriesOutputAdapter(
            num_output_query_channels=latent_channels,
            num_output_channels=num_input_channels,
        )

        self.decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=query_provider,
            num_latent_channels=latent_channels,
            # The decoder's cross-attention also needs the head parameter
            num_cross_attention_heads=num_cross_attention_heads, # <-- ADD THIS LINE
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.encoder(x)
        return self.decoder(latents)

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