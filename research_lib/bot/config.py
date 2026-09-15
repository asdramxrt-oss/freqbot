"""Bot configuration - the single source of truth for a bot run.

Loaded from a JSON file (see ``user_data/bot_config.json``) with dataclass defaults for
every field, so a partial config file is valid. Nothing here is exchange-specific except
the ``exchange`` field, which only selects the data directory / ccxt id.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from research_lib.constants import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Sub-configs (mirror the research dataclasses so defaults stay in one place)
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    pairs: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
    timeframe: str = "15m"
    exchange: str = "binance"
    data_dir: str | None = None          # None -> research_lib.constants.RESEARCH_DATA_DIR
    use_information_bars: bool = False    # True -> resample raw OHLCV into information bars
    bar_type: str = "time"                # time | volume | dollar | imbalance
    bar_threshold: float | None = None    # target volume/dollar per information bar
    warmup_bars: int = 5000               # bars of history required before trading
    use_hurst: bool = False               # rolling DFA Hurst is slow (~1 min / 120k bars)


@dataclass
class StrategyConfig:
    # Turtle (trend book)
    turtle_entry_n: int = 20
    turtle_atr_n: int = 20
    turtle_stop_mult: float = 2.0
    turtle_pyramid_step_n: float = 0.5
    turtle_max_units: int = 4
    turtle_risk_per_unit: float = 0.01
    turtle_allow_short: bool = False
    # OU (mean-reversion book)
    ou_z_entry: float = 2.0
    ou_z_exit: float = 0.0
    ou_half_life_timeout_mult: float = 2.5
    ou_min_sample: int = 120
    ou_lookback: int = 480                # rolling window for the OU fit (bars)
    # Pairs book (market-neutral spread)
    pairs_enabled: bool = False
    pairs_a: str = "ETH/USDT"
    pairs_b: str = "BTC/USDT"
    pairs_window: int = 384               # causal local-mean window for the z-score
    pairs_z_entry: float = 2.0
    pairs_z_exit: float = 0.0
    pairs_max_hold_bars: int = 96


@dataclass
class RouterConfig:
    """Regime router + vol-spike kill switch (PDF: Regime Engine -> Signal Router)."""
    hmm_states: int = 3
    hmm_refit_bars: int = 1344            # refit HMM every ~14 days of 15m bars
    hmm_min_train_bars: int = 4000
    vol_spike_horizon_bars: int = 96
    vol_spike_percentile: float = 0.90
    vol_kill_p: float = 0.80              # HALT entries when P(spike) > this
    vol_refit_bars: int = 2688
    turtle_regimes: list[str] = field(default_factory=lambda: ["TREND"])
    pairs_regimes: list[str] = field(default_factory=lambda: ["LOW_VOL_MR", "NEUTRAL"])
    enable_router: bool = True            # False -> trade the raw signal, no regime gating


@dataclass
class MetaConfig:
    """Triple barrier + meta model + bet sizer (PDF: Meta Model -> Bet Sizer)."""
    enable_meta_model: bool = True        # when False, every primary signal passes at full size
    pt_mult: float = 2.0
    sl_mult: float = 2.0
    vertical_bars: int = 48
    train_fraction: float = 0.6           # chronological split for the meta model
    min_events: int = 120                 # below this, the meta model is skipped (no edge data)
    min_test_auc: float = 0.55            # OOS AUC required from EVERY model before enabling
    min_confidence: float = 0.52          # BetSizer min_confidence
    max_confidence_mult: float = 2.0
    base_risk_per_trade: float = 0.01
    vol_scale_col: str = "atr_20"


@dataclass
class RiskConfig:
    per_trade_risk: float = 0.01
    per_symbol_notional: float = 0.25
    portfolio_notional: float = 1.0
    max_leverage: float = 1.0
    max_position_notional: float = 0.25
    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.06
    max_drawdown_kill: float = 0.15
    max_strategy_fraction: float = 0.6
    max_correlation_exposure: float = 0.8


@dataclass
class ExecutionConfig:
    mode: str = "dry_run"                 # dry_run | paper | live
    equity: float = 100_000.0
    maker_fee: float = 0.0010
    taker_fee: float = 0.0010
    spread_bps: float = 1.0
    slippage_bps: float = 0.5
    post_only: bool = False               # True -> submit post-only (maker) limit orders
    recon_every_bars: int | None = 96     # bars between position reconciliations (None -> off)
    redis_url: str | None = None          # None -> in-process fakeredis/None fallback
    lock_ttl_ms: int = 30_000
    max_bars: int | None = None           # cap bars processed (None -> all available)
    poll_seconds: float = 0.0             # sleep between bars in live mode


@dataclass
class BotConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    name: str = "research_bot"
    version: str = "v1"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys (forward compatible)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path | None = None) -> BotConfig:
    """Load a BotConfig from JSON.

    ``path=None`` uses the default ``user_data/bot_config.json`` when it exists, otherwise
    returns pure defaults. A missing file is not an error: the bot must be runnable with
    zero configuration.
    """
    if path is None:
        candidate = PROJECT_ROOT / "user_data" / "bot_config.json"
        path = candidate if candidate.exists() else None
    if path is None:
        return BotConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    return BotConfig(
        data=_from_dict(DataConfig, raw.get("data", {})),
        strategy=_from_dict(StrategyConfig, raw.get("strategy", {})),
        router=_from_dict(RouterConfig, raw.get("router", {})),
        meta=_from_dict(MetaConfig, raw.get("meta", {})),
        risk=_from_dict(RiskConfig, raw.get("risk", {})),
        execution=_from_dict(ExecutionConfig, raw.get("execution", {})),
        name=raw.get("name", "research_bot"),
        version=raw.get("version", "v1"),
    )
