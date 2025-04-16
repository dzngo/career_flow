import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from tinydb import Query, TinyDB

from extractor.jd_extractor import JDExtractor
from scraper.linkedin_scraper import LinkedInScraper
from utils.llm_loader import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


def run_scraping_pipeline(
    title,
    location,
    max_pages,
    batch_size,
    prompt_dir,
    out_csv="scraped_jobs.csv",
    cache_path="cache/job_cache.json",
    llm_name=None,
    save_raw_job_text=False,
):
    """
    Orchestrates the scraping and job description extraction pipeline.

    This function:
    1. Retrieves LinkedIn job IDs using the specified title and location.
    2. Scrapes individual job descriptions for each ID.
    3. Uses LangChain to detect language, translate (if needed), and extract structured info.
    4. Caches results using TinyDB to avoid re-processing.
    5. Saves the final structured job data to a CSV file.

    Args:
        title (str): Job title to search (e.g., "Data Scientist").
        location (str): Location to search (e.g., "Paris").
        max_pages (int): Number of result pages to scrape (each page = ~25 jobs).
        batch_size (int): Number of job descriptions per batch.
        prompt_dir (str): Directory containing prompt templates.
        out_csv (str): Path to the output CSV file for structured results.
        cache_path (str): File path for the TinyDB cache database. Used to skip already-processed jobs.
        llm: Optional LLM model for extraction.
        save_raw_job_text (bool): Whether to include raw job text in the output CSV.

    """
    llm = get_llm(llm_name)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    scraper = LinkedInScraper(title=title, location=location, max_pages=max_pages, batch_size=batch_size)
    extractor = JDExtractor(prompt_dir, llm=llm)
    db = TinyDB(cache_path)
    db_query = Query()

    results = []

    for i, batch in enumerate(scraper):
        logger.info(f"Processing batch #{i + 1} with {len(batch)} jobs")

        # Split job IDs and texts
        job_ids, job_texts = zip(*batch)

        # Filter out already cached
        ids_to_extract, texts_to_extract = [], []
        for jid, text in zip(job_ids, job_texts):
            if db.contains(db_query.job_id == jid):
                logger.info(f"Cached result for job ID {jid}")
                results.append(db.get(db_query.job_id == jid))
            else:
                ids_to_extract.append(jid)
                texts_to_extract.append(text)

        if not texts_to_extract:
            continue

        try:
            extracted_batch = extractor.extract(texts_to_extract)
            for jid, text, structured in zip(ids_to_extract, texts_to_extract, extracted_batch):
                structured["job_id"] = jid
                if save_raw_job_text:
                    structured["raw_job_text"] = text
                results.append(structured)
                db.insert(structured)
        except Exception as e:
            logger.error(f"Batch #{i + 1} failed: {e}")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df)} job entries to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="Machine learning engineer")
    parser.add_argument("--location", type=str, default="Paris")
    parser.add_argument("--max-pages", type=int, default=600)
    parser.add_argument("--prompt-dir", type=str, default="extractor/prompts")
    parser.add_argument("--output", type=str, default="scraped_jobs.csv")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--cache", type=str, default="cache/job_cache.json")
    parser.add_argument(
        "--llm", type=str, default="gemini-2.0-flash", help="LLM model name (currently only gemini supported)"
    )
    parser.add_argument(
        "--save-raw-job-text", action="store_true", help="Whether to include raw job text in the output CSV"
    )

    args = parser.parse_args()

    run_scraping_pipeline(
        title=args.title,
        location=args.location,
        max_pages=args.max_pages,
        prompt_dir=args.prompt_dir,
        out_csv=args.output,
        cache_path=args.cache,
        batch_size=args.batch_size,
        llm_name=args.llm,
        save_raw_job_text=args.save_raw_job_text,
    )
