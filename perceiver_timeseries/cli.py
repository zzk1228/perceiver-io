# perceiver_timeseries/cli.py
from pytorch_lightning import seed_everything
from pytorch_lightning.cli import LightningCLI
from perceiver_timeseries.datamodule import CSVDataModule
from perceiver_timeseries.model import MultivariatePerceiver

if __name__ == "__main__":
    seed_everything(42, workers=True)
    LightningCLI(
        model_class=MultivariatePerceiver,
        datamodule_class=CSVDataModule,
        save_config_callback=None   # 可选：移除默认保存 yaml
    )
