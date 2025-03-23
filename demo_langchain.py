import argparse
import os
from pprint import pprint

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="JD information extraction demo")
    parser.add_argument(
        "--prompt_extraction_path",
        type=str,
        default="prompts/prompt_jd_extraction_template.txt",
        help="Path to prompt template",
    )
    parser.add_argument(
        "--prompt_lang_detection_path",
        type=str,
        default="prompts/language_detection_template",
        help="Path to prompt template",
    )
    parser.add_argument(
        "--prompt_translation_path",
        type=str,
        default="prompts/prompt_translate_template.txt",
        help="Path to prompt template",
    )
    parser.add_argument("--jd-path", type=str, required=True, help="Path to job description")

    return vars(parser.parse_args())


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    args = parse_args()

    with open(args["jd_path"], "r", encoding="utf-8") as f:
        jd_text = f.read()

    with open(args["prompt_extraction_path"], "r", encoding="utf-8") as f:
        jd_extraction_prompt = ChatPromptTemplate.from_template(f.read())

    with open(args["prompt_lang_detection_path"], "r", encoding="utf-8") as f:
        jd_lang_detection_prompt = ChatPromptTemplate.from_template(f.read())

    with open(args["prompt_translation_path"], "r", encoding="utf-8") as f:
        jd_translation_prompt = ChatPromptTemplate.from_template(f.read())

    language_chain = jd_lang_detection_prompt | llm | StrOutputParser()
    translate_chain = jd_translation_prompt | llm | StrOutputParser()
    jd_extraction_chain = jd_extraction_prompt | llm | JsonOutputParser()

    detected_language = language_chain.invoke({"text": jd_text}).strip()
    if detected_language.lower() != "english":
        print(f"Detected language: {detected_language}. Translating to English...")
        jd_text = translate_chain.invoke({"text": jd_text}).strip()
    else:
        print("Job description is already in English. Skipping translation.")
    response = jd_extraction_chain.invoke({"text": jd_text})
    pprint(response)
