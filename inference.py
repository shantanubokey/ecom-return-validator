"""
Return Validation Inference Pipeline
Usage: python inference.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.internvl_lora import ReturnValidator
from data.test_cases import TEST_CASES, create_placeholder_image


def run_single(
    delivery_images: list,
    vendor_images:   list,
    metadata:        dict,
    use_lora:        bool = False,
) -> dict:
    validator = ReturnValidator(use_lora=use_lora)
    result    = validator.validate(delivery_images, vendor_images, metadata)
    result    = validator.validate_accept_return(result)   # enforce logic
    return result


def run_batch(test_cases: list, use_lora: bool = False) -> list:
    validator = ReturnValidator(use_lora=use_lora)
    validator.load()
    results = []
    for case in test_cases:
        print(f"\n[{case['id']}] {case['description']}")
        result = validator.validate(
            case["delivery_images"],
            case["vendor_images"],
            case["metadata"],
        )
        result = validator.validate_accept_return(result)
        results.append(result)
        print(f"  → accept_return: {result['accept_return'].upper()}")
        print(f"  → {json.dumps(result, indent=4)}")
    return results


if __name__ == "__main__":
    create_placeholder_image()

    # Demo with mock predictions (no GPU needed for testing)
    print("\n[Demo] Running with mock predictions (no model load)")
    print("       Set use_lora=True and provide real images for production\n")

    from evaluation.metrics import full_report

    # Simulate model predictions matching ground truth for demo
    mock_predictions = [case["ground_truth"] for case in TEST_CASES]
    ground_truths    = [case["ground_truth"] for case in TEST_CASES]

    full_report(mock_predictions, ground_truths)
