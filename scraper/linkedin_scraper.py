import json
import os
import time
from typing import Iterator, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from utils.html_utils import parse_html_to_text
from utils.logger import get_logger

logger = get_logger(__name__)


class LinkedInScraper:
    """
    Scrapes job postings from LinkedIn based on job title and location.
    Retrieves job IDs, fetches detailed job descriptions, and yields batches of job texts.
    """

    def __init__(
        self,
        title: str,
        location: str,
        max_pages: int = 1,
        delay: int = 1,
        batch_size: int = 5,
        raw_cache_path: Optional[str] = None,
        load_from_cache: bool = False,
    ):
        """
        Initialize the LinkedInScraper.

        Args:
            title (str): Job title to search for.
            location (str): Job location.
            max_pages (int): Number of pages to paginate through.
            delay (int): Delay between requests (in seconds).
            batch_size (int): Number of job descriptions per batch.
            raw_cache_path (Optional[str]): Path to cache file for job descriptions.
            load_from_cache (bool): Whether to load job descriptions from cache only, skip live fetch.
        """
        self.title = title
        self.location = location
        self.max_pages = max_pages
        self.delay = delay
        self.batch_size = batch_size
        self.raw_cache_path = raw_cache_path
        self.load_from_cache = load_from_cache
        self.job_ids = []
        self.job_pairs = []

    def __len__(self) -> int:
        """
        Return the number of job IDs fetched.

        Returns:
            int: Number of job postings found.
        """
        return len(self.job_pairs)

    def __iter__(self) -> Iterator[List[Tuple[str, str]]]:
        """
        Iterates over batches of job descriptions.

        Yields:
            List[Tuple[str, str]]: A batch of (job_id, job_text) tuples.
        """
        for i in range(0, len(self.job_pairs), self.batch_size):
            yield self.job_pairs[i : i + self.batch_size]

    def start_scraping(self):
        """
        Start scraping job descriptions from LinkedIn or load from cache if enabled.
        """
        # If load_from_cache is requested and cache exists, load all and skip live fetch
        if self.load_from_cache and self.raw_cache_path and os.path.exists(self.raw_cache_path):
            logger.info(f"Loading all job texts from cache (skip live fetching): {self.raw_cache_path}")
            with open(self.raw_cache_path, "r", encoding="utf-8") as f:
                try:
                    self.job_pairs = json.load(f)
                    return
                except Exception:
                    logger.warning("Failed to parse raw cache; proceeding with live scraping.")

        # Load existing cache into a lookup for reuse of individual entries
        cached_dict = {}
        if self.raw_cache_path and os.path.exists(self.raw_cache_path):
            logger.info(f"Loading existing raw cache for reuse: {self.raw_cache_path}")
            with open(self.raw_cache_path, "r", encoding="utf-8") as f:
                try:
                    cached_list = json.load(f)
                    cached_dict = {jid: text for jid, text in cached_list}
                except Exception:
                    logger.warning("Failed to parse raw cache; proceeding without reuse.")

        # Fetch all job IDs
        logger.info("Starting live scrape: retrieving job IDs...")
        self.job_ids = self.get_job_ids()
        logger.info(f"Retrieved {len(self.job_ids)} job IDs. Processing descriptions...")

        # Reuse or fetch each job description
        self.job_pairs = []
        for job_id in self.job_ids:
            if job_id in cached_dict:
                logger.info(f"Reusing cached text for job ID: {job_id}")
                self.job_pairs.append((job_id, cached_dict[job_id]))
            else:
                logger.info(f"Fetching description for job ID: {job_id}")
                job_text = self.fetch_job_description(job_id)
                if job_text:
                    self.job_pairs.append((job_id, job_text))

        if self.raw_cache_path:
            os.makedirs(os.path.dirname(self.raw_cache_path), exist_ok=True)
            with open(self.raw_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.job_pairs, f, indent=2, ensure_ascii=False)

    def get_job_ids(self) -> List[str]:
        """
        Retrieve job posting IDs from LinkedIn search results.

        Returns:
            List[str]: List of job posting IDs.
        """
        job_ids = []
        start = 0
        while start < self.max_pages:
            url = (
                f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?"
                f"keywords={self.title}&location={self.location}&start={start}"
            )
            logger.info(f"Fetching job list from: {url}")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            jobs = soup.find_all("li")

            for job in jobs:
                div = job.find("div", {"class": "base-card"})
                if div and div.get("data-entity-urn"):
                    job_id = div.get("data-entity-urn").split(":")[-1]
                    job_ids.append(job_id)
            time.sleep(self.delay)
            start += len(jobs)
        logger.info(f"Found {len(job_ids)} job IDs")
        return job_ids

    def fetch_job_description(self, job_id: str) -> Optional[str]:
        """
        Fetch the job description and salary for a given job ID.

        Args:
            job_id (str): LinkedIn job ID.

        Returns:
            Optional[str]: Full job text (salary + description), or None if not found.
        """
        url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        try:
            salary = soup.find("div", {"class": "salary"}).text.strip()
        except Exception:
            salary = "Not specified"

        try:
            desc_div = soup.find("div", {"class": "show-more-less-html__markup"})
            if not desc_div:
                return None
            desc = parse_html_to_text(desc_div)
        except Exception:
            return None

        # Combine salary and description with proper newlines
        return f"Salary: {salary}.\n\nDescription:\n{desc}"
