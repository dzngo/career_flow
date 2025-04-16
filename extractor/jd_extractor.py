import os
from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.prompt_loader import load_prompt


class JDExtractor:
    """
    Extracts structured job information using LangChain and prompt templates.
    """

    def __init__(self, prompt_dir: str, llm=None):
        """
        Initialize JDExtractor with prompt directory and optional LLM model.

        Args:
            prompt_dir (str): Path to the directory containing prompt templates.
            llm (BaseLanguageModel, optional): Custom LLM instance. Defaults to Gemini 2.0 Flash.
        """
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.prompts = self._load_prompts(prompt_dir)

    def _load_prompts(self, prompt_dir: str) -> Dict[str, object]:
        """
        Load all prompt templates from the directory.
        Returns:
            dict[str, ChatPromptTemplate]: Dictionary of prompt templates.
        """
        return {
            "extract": load_prompt(os.path.join(prompt_dir, "jd_extraction_batching.txt")),
            "translate": load_prompt(os.path.join(prompt_dir, "translation_batching.txt")),
        }

    def format_jobs_for_batching(self, job_texts: List[str]) -> str:
        """
        Wrap job descriptions with batch delimiters.
        Args:
            job_texts (list[str]): List of job description texts.
        Returns:
            str: Combined formatted batch text.
        """
        return "\n\n".join(f"### JOB START ###\n{text.strip()}\n### JOB END ###" for text in job_texts)

    def extract(self, job_texts: List[str]) -> List[Dict]:
        """
        Perform translation and extraction on a batch of job descriptions.
        Args:
            job_texts (list[str]): List of raw job descriptions.
        Returns:
            list[dict]: Extracted structured results for each job.
        """
        batched_text = self.format_jobs_for_batching(job_texts)
        translate_chain = self.prompts["translate"] | self.llm | StrOutputParser()
        extract_chain = self.prompts["extract"] | self.llm | JsonOutputParser()
        full_chain = translate_chain | extract_chain

        return full_chain.invoke({"text": batched_text})
