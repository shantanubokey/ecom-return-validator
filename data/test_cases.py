"""
Synthetic test cases for return validation evaluation.
Each case has: delivery_images, vendor_images, metadata, ground_truth labels.
In production replace image paths with real delivery/vendor photos.
"""

from pathlib import Path

# Placeholder image path — replace with real images
PLACEHOLDER = str(Path(__file__).parent / "placeholder.jpg")


TEST_CASES = [
    # ── CASE 1: Legitimate return — exact match, unused, undamaged ────────────
    {
        "id": "TC001",
        "description": "Valid return — correct product, unused",
        "delivery_images": [PLACEHOLDER] * 4,
        "vendor_images":   [PLACEHOLDER] * 4,
        "metadata": {
            "product":  "Blue Denim Jacket",
            "brand":    "Levi's",
            "color":    "blue",
            "design":   "plain denim",
            "quantity": "1",
        },
        "ground_truth": {
            "product_match":   "yes",
            "design_match":    "yes",
            "color_match":     "yes",
            "quantity_is_one": "yes",
            "is_damaged":      "no",
            "is_used":         "no",
            "accept_return":   "yes",
        },
    },

    # ── CASE 2: Wrong product returned (fraud) ────────────────────────────────
    {
        "id": "TC002",
        "description": "Fraud — different product returned",
        "delivery_images": [PLACEHOLDER] * 4,
        "vendor_images":   [PLACEHOLDER] * 4,
        "metadata": {
            "product":  "Red Silk Saree",
            "brand":    "Fabindia",
            "color":    "red",
            "design":   "zari border",
            "quantity": "1",
        },
        "ground_truth": {
            "product_match":   "no",
            "design_match":    "no",
            "color_match":     "no",
            "quantity_is_one": "yes",
            "is_damaged":      "no",
            "is_used":         "no",
            "accept_return":   "no",
        },
    },

    # ── CASE 3: Damaged product ───────────────────────────────────────────────
    {
        "id": "TC003",
        "description": "Reject — product is damaged",
        "delivery_images": [PLACEHOLDER] * 4,
        "vendor_images":   [PLACEHOLDER] * 4,
        "metadata": {
            "product":  "Ceramic Coffee Mug",
            "brand":    "Borosil",
            "color":    "white",
            "design":   "plain",
            "quantity": "1",
        },
        "ground_truth": {
            "product_match":   "yes",
            "design_match":    "yes",
            "color_match":     "yes",
            "quantity_is_one": "yes",
            "is_damaged":      "yes",
            "is_used":         "no",
            "accept_return":   "no",
        },
    },

    # ── CASE 4: Used product returned ─────────────────────────────────────────
    {
        "id": "TC004",
        "description": "Reject — product is used",
        "delivery_images": [PLACEHOLDER] * 4,
        "vendor_images":   [PLACEHOLDER] * 4,
        "metadata": {
            "product":  "Running Shoes",
            "brand":    "Nike",
            "color":    "black",
            "design":   "air max",
            "quantity": "1",
        },
        "ground_truth": {
            "product_match":   "yes",
            "design_match":    "yes",
            "color_match":     "yes",
            "quantity_is_one": "yes",
            "is_damaged":      "no",
            "is_used":         "yes",
            "accept_return":   "no",
        },
    },

    # ── CASE 5: Wrong color ───────────────────────────────────────────────────
    {
        "id": "TC005",
        "description": "Reject — color mismatch",
        "delivery_images": [PLACEHOLDER] * 4,
        "vendor_images":   [PLACEHOLDER] * 4,
        "metadata": {
            "product":  "Cotton Kurta",
            "brand":    "W",
            "color":    "green",
            "design":   "floral print",
            "quantity": "1",
        },
        "ground_truth": {
            "product_match":   "yes",
            "design_match":    "yes",
            "color_match":     "no",
            "quantity_is_one": "yes",
            "is_damaged":      "no",
            "is_used":         "no",
            "accept_return":   "no",
        },
    },

    # ── CASE 6: Multiple units returned (quantity fraud) ──────────────────────
    {
        "id": "TC006",
        "description": "Reject — quantity > 1 returned",
        "delivery_images": [PLACEHOLDER] * 4,
        "vendor_images":   [PLACEHOLDER] * 4,
        "metadata": {
            "product":  "Wireless Earbuds",
            "brand":    "boAt",
            "color":    "black",
            "design":   "tws",
            "quantity": "1",
        },
        "ground_truth": {
            "product_match":   "yes",
            "design_match":    "yes",
            "color_match":     "yes",
            "quantity_is_one": "no",
            "is_damaged":      "no",
            "is_used":         "no",
            "accept_return":   "no",
        },
    },
]


def create_placeholder_image():
    """Create a grey placeholder image for testing without real images."""
    from PIL import Image, ImageDraw, ImageFont
    import os
    path = str(Path(__file__).parent / "placeholder.jpg")
    if not os.path.exists(path):
        img = Image.new("RGB", (448, 448), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 398, 398], outline=(100, 100, 100), width=3)
        draw.text((150, 200), "TEST IMAGE", fill=(80, 80, 80))
        img.save(path)
        print(f"[Data] Placeholder image created: {path}")
    return path
