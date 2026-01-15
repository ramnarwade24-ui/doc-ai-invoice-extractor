from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
	import layoutparser as lp  # type: ignore
except Exception:  # pragma: no cover
	lp = None


class _FallbackRectangle:
	def __init__(self, x_1: int, y_1: int, x_2: int, y_2: int):
		self.x_1 = float(x_1)
		self.y_1 = float(y_1)
		self.x_2 = float(x_2)
		self.y_2 = float(y_2)


class _FallbackTextBlock:
	def __init__(self, block: _FallbackRectangle, text: str = "", score: float = 0.0):
		self.block = block
		self.text = text
		self.score = score


class _FallbackLayout(list):
	pass


def _Rectangle(x1: int, y1: int, x2: int, y2: int):
	global lp
	if lp is not None:
		try:
			return lp.Rectangle(x1, y1, x2, y2)
		except Exception:
			lp = None
	return _FallbackRectangle(x1, y1, x2, y2)


def _TextBlock(rect, text: str, score: float):
	global lp
	if lp is not None:
		try:
			return lp.TextBlock(rect, text=text, score=score)
		except Exception:
			lp = None
	return _FallbackTextBlock(rect, text=text, score=score)


def _Layout(items):
	global lp
	if lp is not None:
		try:
			return lp.Layout(items)
		except Exception:
			lp = None
	return _FallbackLayout(items)

from ocr import OCRPage, OCRWord
from utils.text import normalize_spaces


@dataclass(frozen=True)
class Line:
	text: str
	bbox: Tuple[int, int, int, int]
	avg_conf: float
	page_index: int
	region: Optional[str] = None

	def center(self) -> Tuple[float, float]:
		x1, y1, x2, y2 = self.bbox
		return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

	def height(self) -> int:
		return int(self.bbox[3] - self.bbox[1])

	def width(self) -> int:
		return int(self.bbox[2] - self.bbox[0])


@dataclass(frozen=True)
class Region:
	name: str  # header/body/table/footer
	bbox: Tuple[int, int, int, int]
	confidence: float
	lines: List[Line]
	blocks: object

	def text(self) -> str:
		return "\n".join([ln.text for ln in self.lines if ln.text])


@dataclass(frozen=True)
class StructuredLayout:
	page_index: int
	page_size: Tuple[int, int]  # (width,height)
	regions: Dict[str, Region]
	all_lines: List[Line]

	def region(self, name: str) -> Region:
		return self.regions[name]

	def get_lines(self, names: List[str]) -> List[Line]:
		out: List[Line] = []
		for n in names:
			reg = self.regions.get(n)
			if reg:
				out.extend(reg.lines)
		return out


def merge_structured_layouts(layouts: List[StructuredLayout]) -> StructuredLayout:
	"""Merge per-page StructuredLayout objects into a single doc-level layout.

	Keeps region buckets by concatenating region lines. BBoxes become the union.
	"""
	if not layouts:
		# Downstream extraction expects standard region keys to exist.
		empty_regions: Dict[str, Region] = {}
		for name in ("header", "body", "table", "footer"):
			empty_regions[name] = Region(name=name, bbox=(0, 0, 0, 0), confidence=0.0, lines=[], blocks=_Layout([]))
		return StructuredLayout(page_index=-1, page_size=(0, 0), regions=empty_regions, all_lines=[])

	def union_bbox(bbs: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
		bbs = [b for b in bbs if b and b != (0, 0, 0, 0)]
		if not bbs:
			return (0, 0, 0, 0)
		x1 = min(b[0] for b in bbs)
		y1 = min(b[1] for b in bbs)
		x2 = max(b[2] for b in bbs)
		y2 = max(b[3] for b in bbs)
		return (x1, y1, x2, y2)

	# Multi-page intelligence: promote numeric-dense body lines to table if tables continue across pages.
	def _numeric_dense(lines: List[Line]) -> List[Line]:
		out: List[Line] = []
		for ln in lines:
			t = ln.text or ""
			digits = sum(ch.isdigit() for ch in t)
			if len(t) >= 6 and (digits / max(1, len(t))) >= 0.25:
				out.append(ln)
		return out

	any_table_pages = {sl.page_index for sl in layouts if sl.regions.get("table") and sl.regions["table"].lines}
	for sl in layouts:
		reg_table = sl.regions.get("table")
		reg_body = sl.regions.get("body")
		if reg_table and reg_body and (not reg_table.lines):
			# If a neighboring page has a table, treat numeric-dense lines as continuation.
			if (sl.page_index - 1 in any_table_pages) or (sl.page_index + 1 in any_table_pages):
				cand = _numeric_dense(reg_body.lines)
				if len(cand) >= 4:
					# Re-tag these lines as table lines for downstream extraction.
					new_table_lines = [
						Line(text=ln.text, bbox=ln.bbox, avg_conf=ln.avg_conf, page_index=ln.page_index, region="table")
						for ln in cand
					]
					layouts[layouts.index(sl)] = StructuredLayout(
						page_index=sl.page_index,
						page_size=sl.page_size,
						regions={
							**sl.regions,
							"table": Region(
								name="table",
								bbox=reg_body.bbox,
								confidence=max(0.5, float(reg_body.confidence)),
								lines=new_table_lines,
								blocks=reg_body.blocks,
							),
						},
						all_lines=sl.all_lines,
					)

	regions: Dict[str, Region] = {}
	for name in ("header", "body", "table", "footer"):
		lines: List[Line] = []
		bbs: List[Tuple[int, int, int, int]] = []
		confs: List[float] = []
		blocks = []
		for sl in layouts:
			reg = sl.regions.get(name)
			if not reg:
				continue
			lines.extend(reg.lines)
			bbs.append(reg.bbox)
			confs.append(float(reg.confidence))
			blocks.extend(list(reg.blocks))
		bbox = union_bbox(bbs)
		conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0
		regions[name] = Region(name=name, bbox=bbox, confidence=conf, lines=lines, blocks=_Layout(blocks))

	all_lines = []
	for sl in layouts:
		all_lines.extend(sl.all_lines)

	# Page size is unknown for merged; keep first page size
	return StructuredLayout(
		page_index=-1,
		page_size=layouts[0].page_size,
		regions=regions,
		all_lines=all_lines,
	)


def _word_bbox_xyxy(word: OCRWord) -> Tuple[int, int, int, int]:
	# PaddleOCR bbox is 4 points; convert to xyxy
	xs = [p[0] for p in word.bbox]
	ys = [p[1] for p in word.bbox]
	x1, x2 = int(min(xs)), int(max(xs))
	y1, y2 = int(min(ys)), int(max(ys))
	return x1, y1, x2, y2


def build_layout_from_ocr(page: OCRPage) -> lp.Layout:
	blocks = []
	for w in page.words:
		x1, y1, x2, y2 = _word_bbox_xyxy(w)
		rect = _Rectangle(x1, y1, x2, y2)
		blocks.append(_TextBlock(rect, text=w.text, score=w.conf))
	return _Layout(blocks)


def group_words_into_lines(page: OCRPage, y_tol: int = 12) -> List[Line]:
	words = []
	for w in page.words:
		x1, y1, x2, y2 = _word_bbox_xyxy(w)
		words.append((y1, x1, x2, y2, w))

	words.sort(key=lambda t: (t[0], t[1]))
	lines: List[List[Tuple[int, int, int, int, OCRWord]]] = []

	for item in words:
		y1 = item[0]
		if not lines:
			lines.append([item])
			continue
		prev_y = lines[-1][0][0]
		if abs(y1 - prev_y) <= y_tol:
			lines[-1].append(item)
		else:
			lines.append([item])

	out: List[Line] = []
	for ln in lines:
		ln.sort(key=lambda t: t[1])
		texts = [t[4].text for t in ln if t[4].text]
		text = normalize_spaces(" ".join(texts))
		x1 = min(t[1] for t in ln)
		y1 = min(t[0] for t in ln)
		x2 = max(t[2] for t in ln)
		y2 = max(t[3] for t in ln)
		confs = [t[4].conf for t in ln]
		avg_conf = float(sum(confs) / max(1, len(confs)))
		if text:
			out.append(Line(text=text, bbox=(x1, y1, x2, y2), avg_conf=avg_conf, page_index=page.page_index))

	return out


def _rect_intersection_over_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)
	if ix2 <= ix1 or iy2 <= iy1:
		return 0.0
	inter = float((ix2 - ix1) * (iy2 - iy1))
	area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
	return inter / area


def detect_regions(
	all_lines: List[Line],
	page_width: int,
	page_height: int,
) -> Dict[str, Tuple[Tuple[int, int, int, int], float, List[Line]]]:
	"""Heuristic region detection (CPU-only).

	- header: top band
	- footer: bottom band
	- body: remaining
	- table: detected within body by lexical cues + numeric density
	"""
	header_y = int(0.18 * page_height)
	footer_y = int(0.82 * page_height)

	header = [ln for ln in all_lines if ln.bbox[3] <= header_y]
	footer = [ln for ln in all_lines if ln.bbox[1] >= footer_y]
	body = [ln for ln in all_lines if ln not in header and ln not in footer]

	def bbox_of(lines: List[Line]) -> Tuple[int, int, int, int]:
		if not lines:
			return (0, 0, page_width, page_height)
		x1 = min(l.bbox[0] for l in lines)
		y1 = min(l.bbox[1] for l in lines)
		x2 = max(l.bbox[2] for l in lines)
		y2 = max(l.bbox[3] for l in lines)
		return (x1, y1, x2, y2)

	regions: Dict[str, Tuple[Tuple[int, int, int, int], float, List[Line]]] = {}
	regions["header"] = (bbox_of(header), 0.85 if header else 0.35, header)
	regions["footer"] = (bbox_of(footer), 0.85 if footer else 0.35, footer)
	regions["body"] = (bbox_of(body), 0.8 if body else 0.4, body)

	# Table detection inside body
	if body:
		table_keywords = (
			"qty",
			"quantity",
			"rate",
			"amount",
			"total",
			"gst",
			"cgst",
			"sgst",
			"igst",
			"hsn",
			"tax",
			"मात्रा",
			"दर",
			"राशि",
			"कुल",
			"જથ્થો",
			"દર",
			"રકમ",
			"કુલ",
		)
		kw_hits = [ln for ln in body if any(k in ln.text.lower() for k in table_keywords)]
		numeric_dense = []
		for ln in body:
			t = ln.text
			digits = sum(ch.isdigit() for ch in t)
			if len(t) >= 6 and (digits / max(1, len(t))) >= 0.25:
				numeric_dense.append(ln)

		candidates = list({*kw_hits, *numeric_dense})
		# Require enough evidence to avoid false positives
		if len(candidates) >= 4:
			table_bbox = bbox_of(candidates)
			conf = 0.75
			regions["table"] = (table_bbox, conf, candidates)
		else:
			regions["table"] = ((0, 0, 0, 0), 0.0, [])
	else:
		regions["table"] = ((0, 0, 0, 0), 0.0, [])

	return regions


def assign_blocks_to_regions(blocks, region_boxes: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, object]:
	out: Dict[str, List[object]] = {"header": [], "body": [], "table": [], "footer": []}

	def as_xyxy(r) -> Tuple[int, int, int, int]:
		return (int(r.x_1), int(r.y_1), int(r.x_2), int(r.y_2))

	for b in blocks or []:
		# Duck-typed TextBlock
		if not hasattr(b, "block"):
			continue
		bb = as_xyxy(getattr(b, "block"))
		# Prefer table if intersection is meaningful
		if region_boxes.get("table") and region_boxes["table"] != (0, 0, 0, 0):
			ioa = _rect_intersection_over_area(bb, region_boxes["table"])
			if ioa >= 0.2:
				out["table"].append(b)
				continue
		# Otherwise assign by center-Y band using region boxes (header/footer/body)
		cx = (bb[0] + bb[2]) / 2.0
		cy = (bb[1] + bb[3]) / 2.0
		if region_boxes.get("header") and cy <= region_boxes["header"][3]:
			out["header"].append(b)
		elif region_boxes.get("footer") and cy >= region_boxes["footer"][1]:
			out["footer"].append(b)
		else:
			out["body"].append(b)

	return {k: _Layout(v) for k, v in out.items()}


def build_structured_layout(page: OCRPage, page_size: Tuple[int, int]) -> StructuredLayout:
	"""Build a structured layout for a page.

	Uses OCR word boxes to create text blocks and groups them into regions.
	"""
	page_width, page_height = page_size
	all_lines = group_words_into_lines(page)
	regions = detect_regions(all_lines, page_width=page_width, page_height=page_height)

	region_boxes = {k: v[0] for k, v in regions.items()}
	blocks = build_layout_from_ocr(page)
	blocks_by_region = assign_blocks_to_regions(blocks, region_boxes)

	region_objs: Dict[str, Region] = {}
	for name in ("header", "body", "table", "footer"):
		bbox, conf, region_lines = regions.get(name, ((0, 0, 0, 0), 0.0, []))
		# mark region on lines
		marked = [
			Line(
				text=ln.text,
				bbox=ln.bbox,
				avg_conf=ln.avg_conf,
				page_index=ln.page_index,
				region=name,
			)
			for ln in region_lines
		]
		region_objs[name] = Region(
			name=name,
			bbox=bbox,
			confidence=float(conf),
			lines=marked,
			blocks=blocks_by_region.get(name, _Layout([])),
		)

	# Include unassigned lines in body as fallback
	if not region_objs["body"].lines and all_lines:
		region_objs["body"] = Region(
			name="body",
			bbox=(0, 0, page_width, page_height),
			confidence=0.4,
			lines=[
				Line(
					text=ln.text,
					bbox=ln.bbox,
					avg_conf=ln.avg_conf,
					page_index=ln.page_index,
					region="body",
				)
				for ln in all_lines
			],
			blocks=blocks_by_region.get("body", _Layout([])),
		)

	return StructuredLayout(
		page_index=page.page_index,
		page_size=(page_width, page_height),
		regions=region_objs,
		all_lines=all_lines,
	)
