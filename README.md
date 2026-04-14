# E-Commerce Return Fraud Detection
## InternVL2.5-4B MPO + LoRA — Multimodal Return Validation

Detects fraudulent product returns by comparing customer-submitted delivery images against vendor reference images and product metadata using a fine-tuned Vision-Language Model.

---

## The Problem

E-commerce platforms lose billions annually to return fraud:
- Customer returns a **different product** (wrong item swap)
- Customer returns a **damaged** product they broke
- Customer returns a **used** product (worn, dirty, missing tags)
- Customer returns **multiple units** for a single-unit order
- Customer returns a **color/design mismatch** (different variant)

Manual review is slow and error-prone. This system automates it with multimodal AI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT                                        │
│                                                                 │
│  4 Delivery Images    4 Vendor Images    Vendor Metadata        │
│  (customer return)    (original product) (product, brand,       │
│                                           color, design, qty)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              InternVL2.5-4B MPO + LoRA                         │
│                                                                 │
│  • Base: InternVL2.5-4B MPO (Vision-Language Model)            │
│  • Fine-tuned: LoRA on ~500K delivery/vendor image pairs        │
│  • Learns: packaged vs dismantled states, real-world variation  │
│  • Input: 8 images (4 delivery + 4 vendor) + text prompt       │
│  • Output: strict JSON with 7 binary fields                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    JSON OUTPUT                                  │
│                                                                 │
│  {                                                              │
│    "product_match":   "yes/no",  ← same product type?          │
│    "design_match":    "yes/no",  ← same pattern/design?        │
│    "color_match":     "yes/no",  ← exact color match?          │
│    "quantity_is_one": "yes/no",  ← exactly 1 unit?             │
│    "is_damaged":      "yes/no",  ← visible damage?             │
│    "is_used":         "yes/no",  ← signs of use?               │
│    "accept_return":   "yes/no"   ← final decision              │
│  }                                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOGIC GATE                                   │
│                                                                 │
│  accept_return = "yes" ONLY IF:                                 │
│    product_match = yes  AND                                     │
│    design_match  = yes  AND                                     │
│    color_match   = yes  AND                                     │
│    quantity_is_one = yes AND                                    │
│    is_damaged    = no   AND                                     │
│    is_used       = no                                           │
│                                                                 │
│  Otherwise: accept_return = "no"                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Details

| Property | Value |
|---|---|
| Base Model | `OpenGVLab/InternVL2_5-4B-MPO` |
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| Training Data | ~500K delivery + vendor image pairs |
| Input Images | 8 total (4 delivery + 4 vendor), 448×448 |
| Output | Strict JSON, binary yes/no fields |
| Quantization | 4-bit (BitsAndBytes) for inference |
| Task Type | Multimodal binary classification |

### Why InternVL2.5-4B MPO?
- Strong vision-language alignment for product understanding
- MPO (Multi-Preference Optimization) improves instruction following
- 4B parameters — runs on a single 16GB GPU with 4-bit quantization
- Handles packaging/assembly differences via LoRA fine-tuning

### Why LoRA?
- Fine-tunes only ~0.1% of parameters — fast and memory efficient
- Preserves base model's general vision understanding
- Adapts to domain-specific patterns: packaging states, Indian product aesthetics, Hinglish descriptions

---

## Fraud Detection Cases

| Case | Scenario | accept_return |
|---|---|---|
| Correct product, unused, undamaged | Legitimate return | yes |
| Wrong product returned | Product swap fraud | no |
| Product is damaged | Customer damaged it | no |
| Product is used | Worn/dirty/tags removed | no |
| Wrong color variant | Color swap fraud | no |
| Multiple units returned | Quantity fraud | no |

---

## Evaluation Metrics

### Per-Field Metrics
Each of the 7 binary fields is evaluated independently:
- **Precision** — of all "yes" predictions, how many were correct
- **Recall** — of all actual "yes" cases, how many were caught
- **F1 Score** — harmonic mean of precision and recall

### Fraud Detection Metrics (accept_return)
| Metric | Description |
|---|---|
| True Positive | Correctly accepted valid return |
| True Negative | Correctly rejected fraudulent return |
| False Positive | Rejected a valid return (customer dissatisfaction) |
| False Negative | Accepted a fraudulent return **(COSTLY — compensation loss)** |
| Fraud Slip Rate | FN / (FN + TN) — % of fraud that slipped through |

### Hallucination Score
Measures model self-consistency — does `accept_return` match the logical outcome of the other 6 fields?

```
Hallucination = model says accept_return=yes
                but (product_match=no OR is_damaged=yes OR ...)

Hallucination Rate = hallucinated_cases / total_cases
Consistency Score  = 1 - hallucination_rate
```

A well-calibrated model should have **hallucination rate < 2%**.

---

## Project Structure

```
fraud_detection_ecommerce/
├── model/
│   └── internvl_lora.py      ← InternVL2.5 + LoRA inference engine
├── data/
│   └── test_cases.py         ← 6 annotated test cases + placeholder generator
├── evaluation/
│   └── metrics.py            ← F1, precision, recall, hallucination, confusion matrix
├── inference.py              ← CLI inference pipeline
├── notebook.ipynb            ← Interactive evaluation + visualizations
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd fraud_detection_ecommerce
pip install -r requirements.txt
```

For real model inference (requires GPU):
```bash
# Download base model (auto via HuggingFace)
# Place LoRA weights in ./lora_weights/

python inference.py
```

---

## Notebook

Open `notebook.ipynb` for:
1. Test case overview
2. Validation logic diagram
3. Mock predictions (no GPU needed)
4. Per-field F1/Precision/Recall bar chart
5. accept_return confusion matrix
6. Hallucination detection and analysis
7. Heatmap: Ground Truth vs Predictions vs Errors
8. Full evaluation report
9. Real model inference template (uncomment when GPU available)

---

## Real Inference Example

```python
from model.internvl_lora import ReturnValidator

validator = ReturnValidator(use_lora=True, load_in_4bit=True)

result = validator.validate(
    delivery_images=[
        'images/delivery/front.jpg',
        'images/delivery/back.jpg',
        'images/delivery/side.jpg',
        'images/delivery/tag.jpg',
    ],
    vendor_images=[
        'images/vendor/front.jpg',
        'images/vendor/back.jpg',
        'images/vendor/side.jpg',
        'images/vendor/tag.jpg',
    ],
    metadata={
        'product':  'Blue Denim Jacket',
        'brand':    "Levi's",
        'color':    'blue',
        'design':   'plain denim',
        'quantity': '1',
    }
)

result = validator.validate_accept_return(result)
print(result)
# {
#   "product_match": "yes",
#   "design_match": "yes",
#   "color_match": "yes",
#   "quantity_is_one": "yes",
#   "is_damaged": "no",
#   "is_used": "no",
#   "accept_return": "yes"
# }
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Vision-Language Model | InternVL2.5-4B MPO |
| Fine-tuning | LoRA via PEFT |
| Quantization | BitsAndBytes 4-bit |
| Image Processing | PIL + torchvision |
| Evaluation | scikit-learn + matplotlib + seaborn |
| Notebook | Jupyter |
