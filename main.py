"""
=====================================
FATOCHECK - Fake News Detection System
Main Entry Point (v2.0.0)
=====================================

This script serves as the main entry point for the Fatocheck project.
It provides:
- CLI utilities for testing and inference
- Development/debugging capabilities
- Quick model testing interface
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from utils.inference import predict_news

# Project imports
from utils.preprocessing import TextPreprocessor
from utils.settings import configure_logging, get_settings

# =====================================
# Configuration
# =====================================

configure_logging()
logger = logging.getLogger(__name__)

_settings = get_settings()
BASE_DIR = _settings.base_dir
MODELS_DIR = _settings.models_dir


# =====================================
# Utility Functions
# =====================================


def check_model_availability():
    """Check if required model artifacts exist."""

    logger.info("Checking model availability...")

    xgboost_path = _settings.xgboost_model_path
    bert_path = _settings.bert_model_path

    available_models = {}

    if xgboost_path.exists():
        available_models["xgboost"] = True
        logger.info(f"✓ XGBoost model found at {xgboost_path}")
    else:
        available_models["xgboost"] = False
        logger.warning(f"✗ XGBoost model NOT found at {xgboost_path}")

    if bert_path.exists():
        available_models["bert"] = True
        logger.info(f"✓ BERT model found at {bert_path}")
    else:
        available_models["bert"] = False
        logger.warning(f"✗ BERT model NOT found at {bert_path}")

    return available_models


def predict_single(text: str, model_type: Optional[str] = None) -> dict:
    """
    Predict a single news article.

    Args:
        text: News article text
        model_type: Either "xgboost" or "bert" (defaults to MODEL_TYPE / xgboost)

    Returns:
        dict: Prediction result with label and confidence
    """

    if model_type is None:
        model_type = get_settings().default_model_type

    try:
        logger.info(f"Running prediction with model: {model_type}")
        result = predict_news(text, model_type=model_type)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


def predict_batch(texts: list, model_type: Optional[str] = None) -> list:
    """
    Predict multiple news articles.

    Args:
        texts: List of news article texts
        model_type: Either "xgboost" or "bert" (defaults to MODEL_TYPE / xgboost)

    Returns:
        list: List of prediction results
    """

    if model_type is None:
        model_type = get_settings().default_model_type

    logger.info(f"Running batch prediction ({len(texts)} articles) with model: {model_type}")

    results = []
    for i, text in enumerate(texts, 1):
        try:
            result = predict_news(text, model_type=model_type)
            result["article_index"] = i
            results.append(result)
        except Exception as e:
            logger.error(f"Error predicting article {i}: {str(e)}")
            results.append({"article_index": i, "error": str(e)})

    return results


def test_preprocessing(text: str) -> dict:
    """Test text preprocessing pipeline."""

    logger.info("Testing text preprocessing...")

    processor = TextPreprocessor()
    cleaned = processor.clean_text(text)

    return {"original": text, "cleaned": cleaned, "original_length": len(text), "cleaned_length": len(cleaned)}


def interactive_mode():
    """Interactive prediction mode."""

    logger.info("Entering interactive mode. Type 'quit' to exit.")
    print("\n" + "=" * 60)
    print("FATOCHECK - Interactive Fake News Detection")
    print("=" * 60)

    available_models = check_model_availability()

    if not any(available_models.values()):
        logger.error("No models available! Please train models first.")
        return

    model_choice = input("\nSelect model (xgboost/bert) [default: xgboost]: ").strip().lower()

    if not model_choice:
        model_choice = "xgboost"

    if model_choice not in ["xgboost", "bert"] or not available_models.get(model_choice):
        logger.error(f"Model '{model_choice}' not available.")
        return

    print("\n" + "-" * 60)
    print("Enter news text (type 'quit' to exit):")
    print("-" * 60 + "\n")

    while True:
        try:
            text = input("Enter news article text: ").strip()

            if text.lower() == "quit":
                logger.info("Exiting interactive mode.")
                break

            if not text:
                print("Please enter some text.\n")
                continue

            print(f"\nProcessing with {model_choice} model...\n")
            result = predict_single(text, model_type=model_choice)

            print("-" * 60)
            print(f"Model: {result.get('model')}")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Processing Time: {result.get('processing_time_seconds')}s")
            print("-" * 60 + "\n")

        except Exception as e:
            logger.error(f"Error: {str(e)}\n")
            continue


# =====================================
# CLI Commands
# =====================================


def cmd_predict(args):
    """Handle predict command."""

    if args.text:
        result = predict_single(args.text, model_type=args.model)
        print(json.dumps(result, indent=2))

    elif args.file:
        if not Path(args.file).exists():
            logger.error(f"File not found: {args.file}")
            return

        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]

        results = predict_batch(texts, model_type=args.model)
        print(json.dumps(results, indent=2))


def cmd_test(args):
    """Handle test command."""

    sample_text = (
        "Breaking news: Scientists discover revolutionary AI model. " "This AI can detect fake news with 99% accuracy!"
    )

    print("\n" + "=" * 60)
    print("FATOCHECK - Model Test")
    print("=" * 60)

    available_models = check_model_availability()

    if available_models.get("xgboost"):
        print("\n[XGBoost Model]")
        try:
            result = predict_single(sample_text, model_type="xgboost")
            print(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"XGBoost test failed: {str(e)}")

    if available_models.get("bert"):
        print("\n[BERT Model]")
        try:
            result = predict_single(sample_text, model_type="bert")
            print(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"BERT test failed: {str(e)}")


def cmd_preprocess(args):
    """Handle preprocess command."""

    if args.text:
        result = test_preprocessing(args.text)
        print(json.dumps(result, indent=2))


def cmd_status(args):
    """Handle status command."""

    print("\n" + "=" * 60)
    print("FATOCHECK - System Status")
    print("=" * 60)

    available_models = check_model_availability()

    print("\nModel Status:")
    for model, available in available_models.items():
        status = "✓ Available" if available else "✗ Not Found"
        print(f"  {model}: {status}")

    print("\nDirectory Structure:")
    print(f"  Base: {BASE_DIR}")
    print(f"  Models: {MODELS_DIR}")


# =====================================
# Main Function
# =====================================


def main():
    """Main entry point with argument parser."""

    parser = argparse.ArgumentParser(
        description="FatoCheck - AI-powered Fake News Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py predict --text "Your news here"
  python main.py predict --file articles.txt --model bert
  python main.py test
  python main.py interactive
  python main.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict fake news")
    predict_parser.add_argument("--text", type=str, help="Single text to predict")
    predict_parser.add_argument("--file", type=str, help="File with texts (one per line)")
    predict_parser.add_argument(
        "--model",
        type=str,
        default=get_settings().default_model_type,
        choices=["xgboost", "bert"],
        help="Model to use (default: from MODEL_TYPE env, else xgboost)",
    )
    predict_parser.set_defaults(func=cmd_predict)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test models")
    test_parser.set_defaults(func=cmd_test)

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Test text preprocessing")
    preprocess_parser.add_argument("--text", type=str, required=True, help="Text to preprocess")
    preprocess_parser.set_defaults(func=cmd_preprocess)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check system status")
    status_parser.set_defaults(func=cmd_status)

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.set_defaults(func=lambda args: interactive_mode())

    # Parse arguments
    args = parser.parse_args()

    # If no command, show help
    if not args.command:
        parser.print_help()
        return

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
