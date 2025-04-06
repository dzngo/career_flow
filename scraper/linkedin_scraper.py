import time

import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger

logger = get_logger(__name__)


class LinkedInScraper:
    """
    Scrapes job postings from LinkedIn based on job title and location.
    """

    def __init__(self, title, location, max_pages=1, delay=1, batch_size=5):
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

    def __len__(self):
        return len(self.job_ids)

    def __iter__(self):
        """
        Iterates over batches of job descriptions.
        Yields:
            list[tuple[job_id, job_text]]
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

    def get_job_ids(self):
        """
        Retrieve job posting IDs from LinkedIn search results.
        Returns:
            list[str]: List of job posting IDs.
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

    def fetch_job_description(self, job_id):
        """
        Fetch the job description and salary for a given job ID.
        Args:
            job_id (str): LinkedIn job ID.
        Returns:
            str or None: Full job text (salary + description), or None if not found.
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
