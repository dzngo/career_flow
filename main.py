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
    raw_cache_path="cache/raw_job_texts.json",
    structured_cache_path="cache/job_cache.json",
    llm_name=None,
    save_raw_job_text=False,
    use_translation=False,
    load_from_cache=False,
):
    """
    Orchestrate scraping, caching, and structured extraction of job descriptions.

    Workflow:
      1. Initialize scraper with raw cache settings.
      2. Fetch or load raw job texts in batches.
      3. Use JDExtractor (with optional translation) to extract structured info.
      4. Cache structured results in TinyDB to avoid re-processing.
      5. Save all extracted entries to a CSV file.

    Args:
        title (str): Job title to search (e.g., "Data Scientist").
        location (str): Job location to search (e.g., "Paris").
        max_pages (int): Maximum paginated pages to retrieve job IDs.
        batch_size (int): Number of job descriptions to process per batch.
        prompt_dir (str): Directory containing LangChain prompt templates.
        out_csv (str): Output CSV path for structured results.
        raw_cache_path (Optional[str]): Path to JSON cache of raw job texts, or None to disable.
        structured_cache_path (Optional[str]): Path to TinyDB JSON cache for structured results, or None to disable.
        llm_name (Optional[str]): Identifier for the LLM model to use (e.g., "gemini-2.0-flash").
        save_raw_job_text (bool): If True, include the original job text in the CSV output.
        use_translation (bool): If True, translate non-English JDs to English before extraction.
        load_from_cache (bool): If True, skip live scraping and load raw texts from cache.
    """
    llm = get_llm(llm_name)

    os.makedirs(os.path.dirname(raw_cache_path), exist_ok=True)
    scraper = LinkedInScraper(
        title=title,
        location=location,
        max_pages=max_pages,
        batch_size=batch_size,
        raw_cache_path=raw_cache_path,
        load_from_cache=load_from_cache,
    )
    scraper.start_scraping()
    extractor = JDExtractor(prompt_dir, llm=llm, use_translation=use_translation)
    db = TinyDB(structured_cache_path)
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
    parser.add_argument("--cache-dir", type=str, default="cache", help="Base directory for all cache files")
    parser.add_argument("--disable-raw-cache", action="store_true", help="Disable caching of raw scraped job texts")
    parser.add_argument(
        "--disable-structured-cache", action="store_true", help="Disable caching of structured extracted jobs"
    )
    parser.add_argument(
        "--llm", type=str, default="gemini-2.0-flash", help="LLM model name (currently only gemini supported)"
    )
    parser.add_argument(
        "--save-raw-job-text", action="store_true", help="Whether to include raw job text in the output CSV"
    )
    parser.add_argument(
        "--use-translation",
        action="store_true",
        help="Whether to translate job descriptions to English before extraction",
    )
    parser.add_argument("--load-from-cache", action="store_true", help="Whether to load job descriptions from cache")

    args = parser.parse_args()

    # Derive cache paths from cache directory and disable flags
    cache_dir = args.cache_dir
    raw_cache = None if args.disable_raw_cache else os.path.join(cache_dir, "raw_job_texts.json")
    structured_cache = None if args.disable_structured_cache else os.path.join(cache_dir, "job_cache.json")

    run_scraping_pipeline(
        title=args.title,
        location=args.location,
        max_pages=args.max_pages,
        prompt_dir=args.prompt_dir,
        out_csv=args.output,
        raw_cache_path=raw_cache,
        structured_cache_path=structured_cache,
        batch_size=args.batch_size,
        llm_name=args.llm,
        save_raw_job_text=args.save_raw_job_text,
        use_translation=args.use_translation,
        load_from_cache=args.load_from_cache,
    )
