"""
Generate UI screenshots for README documentation.
Creates realistic mockups of the 3 app views without needing a running browser.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

BG      = "#0d1b2a"
CARD    = "#162b3e"
ACCENT  = "#00b4d8"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
ORANGE  = "#f39c12"
WHITE   = "#ffffff"
LIGHT   = "#cad3e0"
PURPLE  = "#8e44ad"

os.makedirs("screenshots", exist_ok=True)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


def make_placeholder_product(color=(100, 149, 237), label="PRODUCT"):
    img = Image.new("RGB", (120, 120), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, 115, 115], outline=(255, 255, 255), width=2)
    draw.text((30, 50), label, fill=(255, 255, 255))
    return img


# ── SCREENSHOT 1: Input View ───────────────────────────────────────────────────
def screenshot_input():
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_facecolor(BG)

    # Title bar
    ax.add_patch(FancyBboxPatch((0, 8.5), 16, 0.5, boxstyle="square",
                                facecolor="#0a1520", edgecolor="none"))
    ax.text(0.3, 8.75, "🔍 Return Fraud Validator", color=WHITE,
            fontsize=14, fontweight="bold", va="center")
    ax.text(14, 8.75, "InternVL2.5-4B MPO + LoRA", color=ACCENT,
            fontsize=8, va="center", style="italic")

    # Sidebar
    ax.add_patch(FancyBboxPatch((0, 0), 2.8, 8.5, boxstyle="square",
                                facecolor="#0a1520", edgecolor="none"))
    ax.text(0.2, 8.1, "📦 Vendor Metadata", color=ACCENT,
            fontsize=9, fontweight="bold")
    fields = [("Product", "Blue Denim Jacket"), ("Brand", "Levi's"),
              ("Color", "blue"), ("Design", "plain denim"), ("Quantity", "1")]
    for i, (k, v) in enumerate(fields):
        y = 7.6 - i * 0.55
        ax.text(0.2, y + 0.15, k, color=LIGHT, fontsize=7)
        ax.add_patch(FancyBboxPatch((0.2, y - 0.1), 2.3, 0.28,
                                    boxstyle="round,pad=0.02",
                                    facecolor=CARD, edgecolor=ACCENT, linewidth=0.5))
        ax.text(0.35, y + 0.04, v, color=WHITE, fontsize=7.5)

    ax.text(0.2, 4.5, "⚙️ Settings", color=ACCENT, fontsize=9, fontweight="bold")
    for i, (lbl, on) in enumerate([("Use LoRA weights", False),
                                    ("4-bit quantization", True),
                                    ("Enable result cache", True)]):
        y = 4.1 - i * 0.4
        color = GREEN if on else LIGHT
        ax.text(0.2, y, f"{'●' if on else '○'} {lbl}", color=color, fontsize=7.5)

    # Delivery images panel
    ax.add_patch(FancyBboxPatch((3.0, 4.5), 6.0, 3.7, boxstyle="round,pad=0.05",
                                facecolor=CARD, edgecolor=ACCENT, linewidth=1))
    ax.text(3.2, 8.0, "📸 Delivery Images (Customer Return)", color=WHITE,
            fontsize=9, fontweight="bold")
    colors_d = [(100, 149, 237), (80, 130, 220), (90, 140, 230), (70, 120, 210)]
    for i in range(4):
        x = 3.2 + i * 1.4
        ax.add_patch(FancyBboxPatch((x, 4.7), 1.2, 1.2, boxstyle="round,pad=0.02",
                                    facecolor=tuple(c/255 for c in colors_d[i]),
                                    edgecolor=WHITE, linewidth=0.5))
        ax.text(x + 0.6, 5.3, f"D{i+1}", color=WHITE, ha="center",
                fontsize=8, fontweight="bold")

    # Vendor images panel
    ax.add_patch(FancyBboxPatch((9.2, 4.5), 6.0, 3.7, boxstyle="round,pad=0.05",
                                facecolor=CARD, edgecolor=ORANGE, linewidth=1))
    ax.text(9.4, 8.0, "🏭 Vendor Images (Original Product)", color=WHITE,
            fontsize=9, fontweight="bold")
    colors_v = [(70, 130, 180), (60, 120, 170), (65, 125, 175), (55, 115, 165)]
    for i in range(4):
        x = 9.4 + i * 1.4
        ax.add_patch(FancyBboxPatch((x, 4.7), 1.2, 1.2, boxstyle="round,pad=0.02",
                                    facecolor=tuple(c/255 for c in colors_v[i]),
                                    edgecolor=WHITE, linewidth=0.5))
        ax.text(x + 0.6, 5.3, f"V{i+1}", color=WHITE, ha="center",
                fontsize=8, fontweight="bold")

    # Validate button
    ax.add_patch(FancyBboxPatch((3.0, 3.8), 12.2, 0.5, boxstyle="round,pad=0.05",
                                facecolor=ACCENT, edgecolor="none"))
    ax.text(9.1, 4.05, "🚀  Validate Return", color=WHITE, ha="center",
            fontsize=11, fontweight="bold")

    ax.text(8, 0.3, "View 1 — Input: Upload delivery & vendor images with metadata",
            color=LIGHT, ha="center", fontsize=8, style="italic")

    plt.savefig("screenshots/01_input_view.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("Saved: screenshots/01_input_view.png")


# ── SCREENSHOT 2: Processing View ─────────────────────────────────────────────
def screenshot_processing():
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_facecolor(BG)

    ax.add_patch(FancyBboxPatch((0, 8.5), 16, 0.5, boxstyle="square",
                                facecolor="#0a1520", edgecolor="none"))
    ax.text(0.3, 8.75, "🔍 Return Fraud Validator", color=WHITE,
            fontsize=14, fontweight="bold", va="center")

    # Pipeline steps
    steps = [
        ("1", "Image Preprocessing", "Loading & resizing 8 images\nChecking image cache...", ACCENT, True),
        ("2", "Tokenization", "Building prompt with metadata\nEncoding image tokens...", ORANGE, True),
        ("3", "Model Inference", "InternVL2.5-4B MPO\n4-bit quantized forward pass...", PURPLE, False),
        ("4", "JSON Parsing", "Extracting structured output\nApplying logic gate...", GREEN, False),
    ]

    for i, (num, title, desc, color, done) in enumerate(steps):
        x = 1.5 + i * 3.5
        alpha = 1.0 if done else 0.4
        ax.add_patch(FancyBboxPatch((x, 4.5), 3.0, 2.5, boxstyle="round,pad=0.1",
                                    facecolor=CARD, edgecolor=color,
                                    linewidth=2 if not done else 1, alpha=alpha))
        ax.add_patch(plt.Circle((x + 1.5, 6.7), 0.3, color=color, alpha=alpha))
        ax.text(x + 1.5, 6.7, num, color=WHITE, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + 1.5, 6.1, title, color=color, ha="center",
                fontsize=8, fontweight="bold", alpha=alpha)
        ax.text(x + 1.5, 5.5, desc, color=LIGHT, ha="center",
                fontsize=7, alpha=alpha, linespacing=1.5)
        if done:
            ax.text(x + 1.5, 4.8, "✓ Done", color=GREEN, ha="center", fontsize=8)
        else:
            ax.text(x + 1.5, 4.8, "⟳ Running...", color=ORANGE, ha="center", fontsize=8)

        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 3.5, 5.75), xytext=(x + 3.0, 5.75),
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))

    # Spinner
    ax.add_patch(FancyBboxPatch((4.5, 2.5), 7, 1.5, boxstyle="round,pad=0.1",
                                facecolor=CARD, edgecolor=ACCENT, linewidth=1.5))
    ax.text(8, 3.5, "⟳  Analyzing images...", color=ACCENT, ha="center",
            fontsize=12, fontweight="bold")
    ax.text(8, 3.0, "GPU: 4-bit inference active  |  Image cache: 6/8 hits",
            color=LIGHT, ha="center", fontsize=8)

    ax.text(8, 0.3, "View 2 — Processing: Pipeline stages with GPU optimization",
            color=LIGHT, ha="center", fontsize=8, style="italic")

    plt.savefig("screenshots/02_processing_view.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("Saved: screenshots/02_processing_view.png")


# ── SCREENSHOT 3: Output View ──────────────────────────────────────────────────
def screenshot_output():
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_facecolor(BG)

    ax.add_patch(FancyBboxPatch((0, 8.5), 16, 0.5, boxstyle="square",
                                facecolor="#0a1520", edgecolor="none"))
    ax.text(0.3, 8.75, "🔍 Return Fraud Validator", color=WHITE,
            fontsize=14, fontweight="bold", va="center")

    # Decision banner — REJECTED
    ax.add_patch(FancyBboxPatch((0.3, 7.8), 15.4, 0.55, boxstyle="round,pad=0.05",
                                facecolor="#4a1a1a", edgecolor=RED, linewidth=2))
    ax.text(8, 8.07, "❌  RETURN REJECTED — quantity_is_one = no (Quantity Fraud Detected)",
            color=RED, ha="center", fontsize=11, fontweight="bold")

    # Fields
    fields = [
        ("Product Match",  "YES", GREEN),
        ("Design Match",   "YES", GREEN),
        ("Color Match",    "YES", GREEN),
        ("Quantity = 1",   "NO",  RED),
        ("Is Damaged",     "NO",  GREEN),
        ("Is Used",        "NO",  GREEN),
        ("Accept Return",  "NO",  RED),
    ]
    for i, (label, val, color) in enumerate(fields):
        row = i % 4
        col = i // 4
        x = 0.5 + col * 5.5
        y = 7.2 - row * 0.55
        ax.add_patch(FancyBboxPatch((x, y - 0.18), 5.0, 0.38,
                                    boxstyle="round,pad=0.02",
                                    facecolor=CARD, edgecolor=color, linewidth=0.8))
        ax.text(x + 0.15, y + 0.01, label, color=LIGHT, fontsize=8, va="center")
        ax.text(x + 4.7, y + 0.01, val, color=color, fontsize=8,
                fontweight="bold", va="center", ha="right")

    # JSON output
    ax.add_patch(FancyBboxPatch((11.2, 4.8), 4.5, 2.8, boxstyle="round,pad=0.05",
                                facecolor="#0a1520", edgecolor=ACCENT, linewidth=1))
    json_lines = [
        '{',
        '  "product_match": "yes",',
        '  "design_match":  "yes",',
        '  "color_match":   "yes",',
        '  "quantity_is_one": "no",',
        '  "is_damaged":    "no",',
        '  "is_used":       "no",',
        '  "accept_return": "no"',
        '}',
    ]
    for i, line in enumerate(json_lines):
        color = RED if '"no"' in line and 'quantity' in line else \
                RED if '"no"' in line and 'accept' in line else \
                GREEN if '"yes"' in line else WHITE
        ax.text(11.4, 7.4 - i * 0.28, line, color=color, fontsize=7,
                fontfamily="monospace")

    # Latency metrics
    metrics = [
        ("847 ms", "This Request"),
        ("612 ms", "Avg Total"),
        ("45 ms",  "Preprocessing"),
        ("780 ms", "Inference"),
        ("3",      "Cache Hits"),
    ]
    for i, (val, lbl) in enumerate(metrics):
        x = 0.5 + i * 2.2
        ax.add_patch(FancyBboxPatch((x, 3.5), 2.0, 1.0, boxstyle="round,pad=0.05",
                                    facecolor=CARD, edgecolor="#1e3a52", linewidth=1))
        ax.text(x + 1.0, 4.15, val, color=ACCENT, ha="center",
                fontsize=11, fontweight="bold")
        ax.text(x + 1.0, 3.75, lbl, color=LIGHT, ha="center", fontsize=7)

    # Latency bar chart
    bars_data = [("Preprocessing", 45, "#3498db"),
                 ("Inference", 780, "#e74c3c"),
                 ("Post-proc", 22, "#27ae60")]
    max_val = 780
    for i, (lbl, val, color) in enumerate(bars_data):
        x = 0.5 + i * 3.5
        bar_w = (val / max_val) * 3.0
        ax.add_patch(FancyBboxPatch((x, 2.3), bar_w, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="none", alpha=0.85))
        ax.text(x + bar_w + 0.1, 2.55, f"{val}ms", color=color, fontsize=8)
        ax.text(x, 2.1, lbl, color=LIGHT, fontsize=7.5)

    ax.text(8, 0.3, "View 3 — Output: Validation result, JSON, latency breakdown",
            color=LIGHT, ha="center", fontsize=8, style="italic")

    plt.savefig("screenshots/03_output_view.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("Saved: screenshots/03_output_view.png")


if __name__ == "__main__":
    screenshot_input()
    screenshot_processing()
    screenshot_output()
    print("\nAll screenshots saved to screenshots/")
