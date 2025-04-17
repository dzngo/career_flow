import time
from typing import Iterator, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger

logger = get_logger(__name__)


class LinkedInScraper:
    """
    Scrapes job postings from LinkedIn based on job title and location.
    Retrieves job IDs, fetches detailed job descriptions, and yields batches of job texts.
    """

    def __init__(self, title: str, location: str, max_pages: int = 1, delay: int = 1, batch_size: int = 5):
        """
        Initialize the LinkedInScraper.

        Args:
            title (str): Job title to search for.
            location (str): Job location.
            max_pages (int): Number of pages to paginate through.
            delay (int): Delay between requests (in seconds).
            batch_size (int): Number of job descriptions per batch.
        """
        self.title = title
        self.location = location
        self.max_pages = max_pages
        self.delay = delay
        self.batch_size = batch_size
        self.job_ids = self.get_job_ids()

    def __len__(self) -> int:
        """
        Return the number of job IDs fetched.

        Returns:
            int: Number of job postings found.
        """
        return len(self.job_ids)

    def __iter__(self) -> Iterator[List[Tuple[str, str]]]:
        """
        Iterates over batches of job descriptions.

        Yields:
            List[Tuple[str, str]]: A batch of (job_id, job_text) tuples.
        """
        batch = []
        for job_id in self.job_ids:
            job_text = self.fetch_job_description(job_id)
            if job_text:
                batch.append((job_id, job_text))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

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
            salary = ""

        try:
            desc = soup.find("div", {"class": "show-more-less-html__markup"}).text.strip()
        except Exception:
            return None

        return f"Salary: {salary}. Description: {desc}"
