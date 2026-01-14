from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def generate_architecture_png(out_path: Path) -> Path:
	"""Generate a simple pipeline flow diagram as a PNG."""
	out_path.parent.mkdir(parents=True, exist_ok=True)

	try:
		import matplotlib.pyplot as plt
		import networkx as nx
	except Exception:
		plt = None
		nx = None

	nodes = [
		"PDF",
		"Images",
		"OCR (PaddleOCR / PyMuPDF fallback)",
		"Layout (header/body/table/footer)",
		"Extraction + Normalization",
		"Vision (YOLOv8) optional",
		"Validation",
		"Latency Guard + Cost Engine",
		"Strict JSON Output",
	]

	# Full diagram if deps exist
	if plt is not None and nx is not None:
		g = nx.DiGraph()
		for n in nodes:
			g.add_node(n)

		edges = list(zip(nodes[:-1], nodes[1:]))
		g.add_edges_from(edges)

		plt.figure(figsize=(12, 4))
		pos = nx.spring_layout(g, seed=7)
		nx.draw_networkx_nodes(g, pos, node_color="#0b5fff", node_size=2200, alpha=0.9)
		nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", width=2, edge_color="#333333")
		nx.draw_networkx_labels(g, pos, font_size=9, font_color="white")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(out_path, dpi=200)
		plt.close()
		return out_path

	# Lightweight fallback diagram using PIL only
	width, height = 1400, 360
	img = Image.new("RGB", (width, height), (255, 255, 255))
	d = ImageDraw.Draw(img)
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None

	d.text((20, 15), "DocAI Invoice Extractor — Architecture", fill=(30, 30, 30), font=font)

	left = 40
	top = 70
	box_w = 150
	box_h = 70
	gap = 20
	# Split into two rows for readability
	row1 = nodes[:5]
	row2 = nodes[5:]

	def draw_row(items, y):
		x = left
		centers = []
		for label in items:
			x1, y1 = x, y
			x2, y2 = x + box_w, y + box_h
			d.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=(11, 95, 255), outline=(11, 95, 255))
			# crude wrap
			words = label.split(" ")
			lines = [""]
			for w in words:
				if len(lines[-1] + " " + w) <= 20:
					lines[-1] = (lines[-1] + " " + w).strip()
				else:
					lines.append(w)
			for li, line in enumerate(lines[:3]):
				d.text((x1 + 8, y1 + 10 + li * 14), line, fill=(255, 255, 255), font=font)
			centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
			x += box_w + gap
		return centers

	cent1 = draw_row(row1, top)
	cent2 = draw_row(row2, top + box_h + 70)

	def arrow(a, b):
		d.line((a[0] + box_w // 2, a[1], b[0] - box_w // 2, b[1]), fill=(50, 50, 50), width=3)
		# arrow head
		x, y = b[0] - box_w // 2, b[1]
		d.polygon([(x, y), (x - 10, y - 6), (x - 10, y + 6)], fill=(50, 50, 50))

	# Row1 arrows
	for i in range(len(cent1) - 1):
		arrow(cent1[i], cent1[i + 1])
	# Bridge arrow from end row1 to start row2
	# down then right
	start = (cent1[-1][0] + box_w // 2, cent1[-1][1])
	end = (cent2[0][0] - box_w // 2, cent2[0][1])
	mid1 = (start[0] + 20, start[1])
	mid2 = (mid1[0], end[1])
	d.line((start[0], start[1], mid1[0], mid1[1]), fill=(50, 50, 50), width=3)
	d.line((mid1[0], mid1[1], mid2[0], mid2[1]), fill=(50, 50, 50), width=3)
	d.line((mid2[0], mid2[1], end[0], end[1]), fill=(50, 50, 50), width=3)
	d.polygon([(end[0], end[1]), (end[0] - 10, end[1] - 6), (end[0] - 10, end[1] + 6)], fill=(50, 50, 50))
	# Row2 arrows
	for i in range(len(cent2) - 1):
		arrow(cent2[i], cent2[i + 1])

	img.save(out_path)
	return out_path


if __name__ == "__main__":
	base = Path(__file__).resolve().parent
	print(generate_architecture_png(base / "outputs" / "architecture_diagram.png"))
