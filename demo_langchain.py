import argparse
import os
from pprint import pprint

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="JD information extraction demo")
    parser.add_argument(
        "--prompt_template_path", type=str, default="prompts/prompt_template.txt", help="Path to prompt template"
    )
    parser.add_argument("--jd-path", type=str, required=True, help="Path to job description")

    return vars(parser.parse_args())


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    args = parse_args()
    jd_template_pth = args["prompt_template_path"]
    with open(jd_template_pth, "r", encoding="utf-8") as f:
        jd_template_str = f.read()
    jd_example_pth = args["jd_path"]
    with open(jd_example_pth, "r", encoding="utf-8") as f:
        jd_str = f.read()

    jd_template = ChatPromptTemplate.from_template(jd_template_str)
    output_parser = JsonOutputParser()

    chain = jd_template | llm | output_parser
    response = chain.invoke({"text": jd_str})
    pprint(response)
