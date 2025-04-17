import argparse
import ast

import pandas as pd

from evaluation.jd_evaluator import JDExtractionEvaluator
from extractor.jd_extractor import JDExtractor
from utils.llm_loader import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)


def load_data(csv_path):
    """
    Load job descriptions and ground truth data from CSV.

    Args:
        csv_path (str): Path to the input CSV file.

    Returns:
        tuple: A list of job description texts and a list of corresponding ground truth dictionaries.
    """
    df = pd.read_csv(csv_path)
    assert "raw_job_text" in df.columns, "CSV must contain a 'raw_job_text' column"

    jd_texts = df["raw_job_text"].tolist()
    structured_columns = ["required_experience", "salary", "skills", "education"]  # TODO: make this cleaner
    for col in structured_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})  # from str to dict
    ground_truths = df.drop(columns=["raw_job_text"]).to_dict(orient="records")
    return jd_texts, ground_truths


def evaluate_predictions(
    input_csv, prompt_dir, llm_model, fields, batch_size, save_output=False, output_csv="evaluation_output.csv"
):
    """
    Runs job description extraction and evaluation.

    This function loads job descriptions and their corresponding ground truth from a CSV file,
    extracts structured information using a selected LLM, and evaluates the results using
    NER-style metrics across the specified fields.

    Args:
        input_csv (str): Path to the input CSV file with 'jd_text' and 'ground_truth' columns.
        prompt_dir (str): Path to the directory containing prompt templates.
        llm_model (str): The identifier for the language model to use.
        fields (List[str]): List of field paths to evaluate (e.g., 'title', 'skills.hard_skills').
        batch_size (int): Number of job descriptions to process per batch.
        save_output (bool): Whether to save extracted predictions to CSV.
        output_csv (str): Path to save extracted predictions.
    """
    logger.info("Loading data...")
    jd_texts, ground_truths = load_data(input_csv)

    logger.info(f"Extracting predictions from {len(jd_texts)} job descriptions...")
    llm = get_llm(llm_model)
    extractor = JDExtractor(prompt_dir=prompt_dir, llm=llm)

    predictions = []
    for i in range(0, len(jd_texts), batch_size):
        batch = jd_texts[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}: jobs {i} to {i + len(batch) - 1}")
        batch_predictions = extractor.extract(batch)
        predictions.extend(batch_predictions)

    logger.info("Running evaluation...")
    evaluator = JDExtractionEvaluator(fields=fields)
    metrics = evaluator.evaluate_batch(ground_truths, predictions)
    logger.info("\nEvaluation Metrics:")
    for field, score in metrics.items():
        logger.info(
            f"{field:25} | Precision: {score['precision']:.3f} | Recall: {score['recall']:.3f} | "
            f"F1: {score['f1']:.3f}"
        )

    if save_output:
        pd.DataFrame(predictions).to_csv(output_csv, index=False)
        logger.info(f"Saved extracted predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True, help="CSV file with jd_text and ground_truth columns")
    parser.add_argument("--prompt-dir", type=str, default="extractor/prompts", help="Directory with prompt templates")
    parser.add_argument("--llm", type=str, default="gemini-2.0-flash", help="LLM model name (e.g., gemini-2.0-flash)")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=[
            "skills.hard_skills",
            "skills.soft_skills",
            "skills.nice_to_have",
            "skills.required_languages",
        ],
        help="Fields to evaluate using NER-style metrics",
    )
    parser.add_argument("--batch-size", type=int, default=5, help="Number of job descriptions to process per batch")
    parser.add_argument("--save-output", action="store_true", help="Whether to save extracted predictions to CSV")
    parser.add_argument(
        "--output-csv", type=str, default="jd_extraction_output.csv", help="Path to save extracted predictions"
    )

    args = parser.parse_args()
    evaluate_predictions(
        args.input_csv,
        args.prompt_dir,
        args.llm,
        args.fields,
        args.batch_size,
        save_output=args.save_output,
        output_csv=args.output_csv,
    )
