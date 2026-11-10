"""Project paths and study configuration."""
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "project.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


@dataclass(frozen=True)
class Config:
    start: str
    holdout_months: int
    market_proxy: str
    price_index: str
    sectors: dict[str, str]  # symbol -> sector
    sources: dict[str, str]
    max_missing_share: float
    jump_threshold: float
    survivorship_haircut: float
    stress_loss: float
    asset_classes: dict
    corporate_actions: tuple[dict, ...] = ()

    @property
    def symbols(self) -> list[str]:
        return list(self.sectors)


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sectors = {sym: sector for sector, syms in raw["universe"].items() for sym in syms}
    return Config(
        start=raw["window"]["start"],
        holdout_months=raw["window"]["holdout_months"],
        market_proxy=raw["benchmark"]["market_proxy"],
        price_index=raw["benchmark"]["price_index"],
        sectors=sectors,
        sources=raw["sources"],
        max_missing_share=raw["quality"]["max_missing_share"],
        jump_threshold=raw["quality"]["jump_threshold"],
        survivorship_haircut=raw["investor"]["survivorship_haircut"],
        stress_loss=raw["investor"]["stress_loss"],
        asset_classes=raw["asset_classes"],
        corporate_actions=tuple(raw.get("corporate_actions") or ()),
    )


def yahoo_ticker(symbol: str) -> str:
    """Map an NSE symbol to its Yahoo ticker; index tickers already start with ^."""
    return symbol if symbol.startswith("^") else f"{symbol}.NS"
