import os

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.prompt_loader import load_prompt


class JDExtractor:
    """
    Extracts structured job information using LangChain and prompt templates.
    """

    def __init__(self, prompt_dir):
        """
        Initialize JDExtractor with prompt directory.

        Args:
            prompt_dir (str): Path to the directory containing prompt templates.
        """
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.prompts = self._load_prompts(prompt_dir)

    def _load_prompts(self, prompt_dir):
        """
        Load all prompt templates from the directory.

        Returns:
            dict[str, ChatPromptTemplate]: Dictionary of prompt templates.
        """
        return {
            "extract": load_prompt(os.path.join(prompt_dir, "jd_extraction.txt")),
            "translate": load_prompt(os.path.join(prompt_dir, "translation.txt")),
        }

    def extract(self, jd_text):
        """
        Perform language detection, translation if needed, and structured extraction.

        Args:
            jd_text (str): Raw job description text.

        Returns:
            dict: Extracted structured data from the job description.
        """
        translate_chain = self.prompts["translate"] | self.llm | StrOutputParser()
        extract_chain = self.prompts["extract"] | self.llm | JsonOutputParser()
        full_chain = translate_chain | extract_chain

        return full_chain.invoke({"text": jd_text})
