from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _parse_percent(value):
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    return float(value)


class MacroAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_environment: Literal["进攻", "防守", "中性", "offensive", "defensive", "neutral"]
    suggested_position: float = Field(ge=0, le=100)
    reason: str = Field(min_length=1)

    @field_validator("suggested_position", mode="before")
    @classmethod
    def parse_suggested_position(cls, value):
        return _parse_percent(value)


class StockRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_code: str = Field(min_length=2, max_length=20)
    action: Literal["买入", "卖出", "持有", "buy", "sell", "hold"]
    weight: float = Field(ge=0, le=100)
    reason: str = Field(min_length=1)

    @field_validator("weight", mode="before")
    @classmethod
    def parse_weight(cls, value):
        return _parse_percent(value)

    @field_validator("stock_code")
    @classmethod
    def normalize_stock_code(cls, value):
        code = value.lower().strip()
        # 已有明确前缀（sh/sz/hk/us）→ 原样返回
        if code[:2].isalpha() and code[2:].isdigit():
            return code
        # 6 位纯数字 A 股代码：6 开头 → 上海，0/3 开头 → 深圳
        if code.isdigit() and len(code) == 6:
            if code.startswith("6"):
                return f"sh{code}"
            if code.startswith(("0", "3")):
                return f"sz{code}"
        # 其他格式（港股/美股/指数代码）→ 原样返回
        return code


class RiskOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_code: str = Field(min_length=2, max_length=20)
    direction: Literal["buy", "sell"]
    quantity_percent: float = Field(ge=0, le=100)
    stop_loss_percent: float = Field(ge=0, le=50)

    @field_validator("quantity_percent", "stop_loss_percent", mode="before")
    @classmethod
    def parse_percent_fields(cls, value):
        return _parse_percent(value)

    @field_validator("stock_code")
    @classmethod
    def normalize_stock_code(cls, value):
        code = value.lower().strip()
        # 已有明确前缀（sh/sz/hk/us）→ 原样返回
        if code[:2].isalpha() and code[2:].isdigit():
            return code
        # 6 位纯数字 A 股代码：6 开头 → 上海，0/3 开头 → 深圳
        if code.isdigit() and len(code) == 6:
            if code.startswith("6"):
                return f"sh{code}"
            if code.startswith(("0", "3")):
                return f"sz{code}"
        # 其他格式（港股/美股/指数代码）→ 原样返回
        return code


class RiskReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    adjustments: list[str] = Field(default_factory=list)
    final_orders: list[RiskOrder] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_orders_when_approved(self):
        if self.approved and not self.final_orders:
            raise ValueError("approved=true 时 final_orders 不能为空")
        return self
