import examples.training  # noqa: F401
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from perceiver.model.core import (
    DecoderConfig,
    EncoderConfig,
    FourierPositionEncoding,
    InputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
    PerceiverIOConfig,
    QueryProvider,
    TrainableQueryProvider,
)


class RandomTimeSeriesDataset(Dataset):
    """Dataset generating random time series data."""

    def __init__(self, num_samples: int, input_len: int, pred_len: int, num_features: int):
        self.data = torch.randn(num_samples, input_len + pred_len, num_features)
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx, : self.input_len]
        y = self.data[idx, self.input_len :]
        return x, y


class RandomTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, input_len: int = 16, pred_len: int = 8, num_features: int = 7):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.ds_train = RandomTimeSeriesDataset(32, self.hparams.input_len, self.hparams.pred_len, self.hparams.num_features)
        self.ds_valid = RandomTimeSeriesDataset(8, self.hparams.input_len, self.hparams.pred_len, self.hparams.num_features)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.hparams.batch_size)


@dataclass
class TimeSeriesEncoderConfig(EncoderConfig):
    seq_len: int = 16
    num_channels: int = 32
    num_frequency_bands: int = 32


@dataclass
class TimeSeriesDecoderConfig(DecoderConfig):
    pred_len: int = 8
    num_channels: int = 32


TimeSeriesConfig = PerceiverIOConfig[TimeSeriesEncoderConfig, TimeSeriesDecoderConfig]


class TimeSeriesInputAdapter(InputAdapter):
    def __init__(self, seq_len: int, num_input_channels: int, num_frequency_bands: int):
        pos_enc = FourierPositionEncoding(input_shape=(seq_len,), num_frequency_bands=num_frequency_bands)
        super().__init__(num_input_channels + pos_enc.num_position_encoding_channels())
        self.pos_enc = pos_enc

    def forward(self, x):
        b = x.shape[0]
        pos = self.pos_enc(b)
        return torch.cat([x, pos], dim=-1)


class TimeSeriesOutputAdapter(OutputAdapter):
    def __init__(self, pred_len: int, num_output_query_channels: int, num_channels: int):
        super().__init__()
        self.pred_len = pred_len
        self.linear = torch.nn.Linear(num_output_query_channels, num_channels)

    def forward(self, x):
        return self.linear(x).view(x.shape[0], self.pred_len, -1)


class TimeSeriesQueryProvider(TrainableQueryProvider):
    def __init__(self, pred_len: int, num_query_channels: int, init_scale: float = 0.02):
        super().__init__(num_queries=pred_len, num_query_channels=num_query_channels, init_scale=init_scale)


class TimeSeriesPerceiver(PerceiverIO):
    def __init__(self, config: TimeSeriesConfig):
        input_adapter = TimeSeriesInputAdapter(
            seq_len=config.encoder.seq_len,
            num_input_channels=config.encoder.num_channels,
            num_frequency_bands=config.encoder.num_frequency_bands,
        )
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            **config.encoder.base_kwargs(),
        )
        output_query_provider = TimeSeriesQueryProvider(
            pred_len=config.decoder.pred_len,
            num_query_channels=config.decoder.num_channels,
            init_scale=config.decoder.init_scale,
        )
        output_adapter = TimeSeriesOutputAdapter(
            pred_len=config.decoder.pred_len,
            num_output_query_channels=config.decoder.num_channels,
            num_channels=config.decoder.num_channels,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=config.num_latent_channels,
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)

    def forward(self, x):
        latents = self.encoder(x)
        return self.decoder(latents)


class LitTimeSeries(pl.LightningModule):
    def __init__(self, config: TimeSeriesConfig):
        super().__init__()
        self.model = TimeSeriesPerceiver(config)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    data = RandomTimeSeriesDataModule()
    config = TimeSeriesConfig(
        encoder=TimeSeriesEncoderConfig(seq_len=data.hparams.input_len),
        decoder=TimeSeriesDecoderConfig(pred_len=data.hparams.pred_len),
        num_latents=32,
        num_latent_channels=64,
    )
    lit_model = LitTimeSeries(config)
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    trainer.fit(lit_model, datamodule=data)
