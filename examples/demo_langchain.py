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
        "--extraction_prompt_path",
        type=str,
        default="extractor/prompts/jd_extraction.txt",
        help="Path to prompt template",
    )
    parser.add_argument(
        "--translation_prompt_path",
        type=str,
        default="extractor/prompts/translation.txt",
        help="Path to prompt template",
    )
    parser.add_argument("--jd-path", type=str, required=True, help="Path to job description")

    return vars(parser.parse_args())


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    args = parse_args()

    with open(args["jd_path"], "r", encoding="utf-8") as f:
        jd_text = f.read()

    with open(args["extraction_prompt_path"], "r", encoding="utf-8") as f:
        jd_extraction_prompt = ChatPromptTemplate.from_template(f.read())

    with open(args["translation_prompt_path"], "r", encoding="utf-8") as f:
        jd_translation_prompt = ChatPromptTemplate.from_template(f.read())

    translate_chain = jd_translation_prompt | llm | StrOutputParser()
    jd_extraction_chain = jd_extraction_prompt | llm | JsonOutputParser()
    full_chain = translate_chain | jd_extraction_chain

    response = full_chain.invoke({"text": jd_text})
    pprint(response)
