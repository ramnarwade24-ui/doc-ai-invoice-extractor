from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PresenceBox(BaseModel):
	present: bool
	bbox: List[int] = Field(default_factory=list, description="[x1,y1,x2,y2] if present else []")

	@field_validator("bbox")
	@classmethod
	def bbox_len(cls, v: List[int]) -> List[int]:
		if len(v) not in (0, 4):
			raise ValueError("bbox must have length 0 or 4")
		return v


class InvoiceFields(BaseModel):
	dealer_name: Optional[str] = None
	model_name: Optional[str] = None
	horse_power: Optional[int] = None
	asset_cost: Optional[int] = None
	signature: PresenceBox = Field(default_factory=lambda: PresenceBox(present=False, bbox=[]))
	stamp: PresenceBox = Field(default_factory=lambda: PresenceBox(present=False, bbox=[]))


class InvoiceOutput(BaseModel):
	doc_id: str
	fields: InvoiceFields
	confidence: float = Field(ge=0.0, le=1.0)
	review_required: bool = False
	processing_time_sec: float = Field(ge=0.0)
	cost_estimate_usd: float = Field(ge=0.0)
	cost_breakdown_usd: dict[str, float] | None = None
