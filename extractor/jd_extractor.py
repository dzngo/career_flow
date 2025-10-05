import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator

from utils.llm_loader import BaseLLM, get_llm
from utils.prompt_loader import load_prompt


class ExperienceYears(BaseModel):
    """Years of experience requirement."""

    min: int = Field(description="Minimum required years of experience. Use -1 when unspecified.")
    max: int = Field(description="Maximum expected years of experience. Use -1 when unspecified.")


class RequiredExperience(BaseModel):
    """Experience requirement structure."""

    years: ExperienceYears
    level: str = Field(description="Experience level such as Internship, Entry, Junior, Mid, or Senior.")


class Salary(BaseModel):
    """Salary range details."""

    min: int = Field(description="Minimum salary value or -1 when unspecified.")
    max: int = Field(description="Maximum salary value or -1 when unspecified.")
    currency: str = Field(description="Salary currency code, empty string when unknown.")


class Skills(BaseModel):
    """Skills grouping."""

    hard_skills: List[str] = Field(description="List of mandatory technical skills.")
    soft_skills: List[str] = Field(description="List of expected soft skills.")
    required_languages: List[str] = Field(
        description="List of required working languages  (e.g., English, French, German). If it does not "
        "explicitly mention required languages, infer the ** original language of the job post as the required "
        "language **. If a language is listed as 'a plus', include it in nice_to_have, not required_languages..",
    )
    nice_to_have: List[str] = Field(description="List of optional or nice-to-have skills.")

    @classmethod
    @field_validator("hard_skills", "soft_skills", "required_languages", "nice_to_have", mode="before")
    def _normalize_list(cls, value):  # noqa: D401 - simple normalization helper
        """Normalize incoming values to a clean list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
            return [item for item in items if item]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []


class Education(BaseModel):
    """Education requirements."""

    degrees: List[str] = Field(description="List of required degrees or certifications.")
    fields_of_study: List[str] = Field(description="List of academic fields required for the role.")

    @classmethod
    @field_validator("degrees", "fields_of_study", mode="before")
    def _normalize_list(cls, value):  # noqa: D401 - simple normalization helper
        """Normalize incoming values to a clean list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
            return [item for item in items if item]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []


class JobExtraction(BaseModel):
    """Structured representation of a single job description."""

    title: str = Field(description="Job title such as Software Engineer or Data Scientist.")
    industry: str = Field(description="High-level industry categorization (e.g., Tech, Finance).")
    employment_type: str = Field(description="Employment type, for example Full-time or Contract.")
    employment_contract: str = Field(description="Contract style such as Permanent, Fixed-term, or Freelance.")
    required_experience: RequiredExperience = Field(
        description="Experience requirements inferred from the job description."
    )
    salary: Salary = Field(description="Salary range details including min, max, and currency.")
    skills: Skills = Field(description="Grouped skill requirements covering hard, soft, language, and optional skills.")
    education: Education = Field(description="Expected education level and relevant fields of study.")
    responsibilities: str = Field(description="Quoted responsibilities or missions extracted from the posting.")
    tech_stack: str = Field(description="Quoted technologies or tooling mentioned for the role.")


class JobExtractionList(RootModel[List[JobExtraction]]):
    """Container for a list of job extraction results."""

    def to_list(self) -> List[JobExtraction]:
        """Expose the underlying list of job extraction results."""

        return list(self.root)


class JDExtractor:
    """Extract structured job information using a configurable LLM backend."""

    def __init__(self, llm: Optional[BaseLLM] = None, use_translation: bool = False):
        """Initialize the extractor with prompt templates and an OpenAI client."""

        self.llm: BaseLLM = llm or get_llm()
        self.use_translation = use_translation

    def _load_prompts(self, prompt_dir: str) -> Dict[str, str]:
        """Load all prompt templates from the directory."""

        return {
            "extract": load_prompt(os.path.join(prompt_dir, "jd_extraction_batching.txt")),
            "translate": load_prompt(os.path.join(prompt_dir, "translation_batching.txt")),
        }

    @staticmethod
    def format_jobs_for_batching(job_texts: List[str]) -> str:
        """Wrap job descriptions with batch delimiters recognized by the prompts."""

        return "\n\n".join(f"### JOB START ###\n{text.strip()}\n### JOB END ###" for text in job_texts)

    def extract(self, job_texts: List[str]) -> List[Dict]:
        """Perform optional translation and structured extraction on job descriptions."""

        batched_text = self.format_jobs_for_batching(job_texts)

        if self.use_translation:
            batched_text = self._translate_batch(batched_text)

        extraction = self._extract_structured_batch(batched_text)
        return [job.model_dump() for job in extraction.to_list()]

    def _translate_batch(self, batched_text: str) -> str:
        """Translate a batch of job descriptions into English using the configured model."""

        prompt = f"Given following job descriptions: {batched_text}"
        return self.llm.complete_text(
            [
                {
                    "role": "system",
                    "content": "You are a professional translator specialized in job descriptions."
                    "For the following job description:"
                    "- First, detect the original language."
                    "- If the job description is **already in English**, return the original text unchanged."
                    "- Otherwise, translate it into English while preserving its structure and meaning."
                    "At the beginning of the output, include this line:"
                    "'The original is written in <DetectedLanguage>. Here is the translated version in English:'",
                },
                {"role": "user", "content": prompt},
            ]
        )

    def _extract_structured_batch(self, batched_text: str) -> JobExtractionList:
        """Call the LLM to extract structured data and validate via Pydantic."""

        prompt = f"Given following job descriptions: {batched_text}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You will extract structured job information from multiple job descriptions. "
                    "Each job is wrapped between ### JOB START ### and ### JOB END ###."
                    "IMPORTANT: The extracted information must be in English, even if the original job description "
                    "is in another language.**"
                    "Return a JSON list with one object per job."
                    "Ensure that: - Elements are listed separately and not combined using or. "
                    "Especially for **skills**."
                    "- Do not include grouped or nested items. Instead, list each item separately. "
                    "Especially for **skills**."
                    "- **Hard_skills** vs **Nice_to_have**: If the job description lists a primary skill "
                    "and then mentions alternatives, include the primary skill in hard_skills and the alternatives "
                    "in nice_to_have. If a list of technologies is given without explicitly saying they are required "
                    "(e.g., “Technologies we use, tech stack, ...”), include them under `nice_to_have` instead."
                    "- If the job description explicitly mentions 'research experience', 'publications', or similar: "
                    "If phrased as required or expected, include 'research"
                    "experience' in hard_skills. If mentioned as optional or nice-to-have, include it in nice_to_have."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            return self.llm.parse_structured(messages, JobExtractionList)
        except Exception:
            raw_text = self.llm.complete_text(messages)
            try:
                return JobExtractionList.model_validate_json(raw_text)
            except ValidationError as validation_error:
                raise RuntimeError("LLM response could not be parsed into the expected schema") from validation_error
