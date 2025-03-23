import argparse
import os
import time

import pandas as pd
from dotenv import load_dotenv
from tinydb import Query, TinyDB

from extractor.jd_extractor import JDExtractor
from scraper.linkedin_scraper import LinkedInScraper
from utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


def run_scraping_pipeline(
    title, location, max_pages, prompt_dir, out_csv="scraped_jobs.csv", cache_path="cache/job_cache.json"
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
        prompt_dir (str): Directory containing prompt templates.
        out_csv (str): Path to the output CSV file for structured results.
        cache_path (str): File path for the TinyDB cache database. Used to skip already-processed jobs.

    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    scraper = LinkedInScraper(title, location, max_pages)
    extractor = JDExtractor(prompt_dir)
    db = TinyDB(cache_path)
    Job = Query()

    job_ids = scraper.get_job_ids()
    logger.info(f"Found {len(job_ids)} job IDs")

    results = []

    for job_id in job_ids:
        logger.info(f"Processing job ID: {job_id}")
        if db.contains(Job.job_id == job_id):
            logger.info(f"Cached result found for job ID: {job_id}")
            results.append(db.get(Job.job_id == job_id))
            continue

        jd_text = scraper.fetch_job_description(job_id)
        if jd_text:
            try:
                data = extractor.extract(jd_text)
                data["job_id"] = job_id
                results.append(data)
                db.insert(data)
            except Exception as e:
                logger.error(f"Failed to extract job {job_id}: {e}")
        else:
            logger.warning(f"No description found for job ID {job_id}")

        time.sleep(1)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    logger.info(f"\nSaved {len(df)} job entries to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="Machine learning engineer")
    parser.add_argument("--location", type=str, default="Paris")
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--prompt-dir", type=str, default="extractor/prompts")
    parser.add_argument("--output", type=str, default="scraped_jobs.csv")
    args = parser.parse_args()

    run_scraping_pipeline(
        title=args.title,
        location=args.location,
        max_pages=args.max_pages,
        prompt_dir=args.prompt_dir,
        out_csv=args.output,
    )
