import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

jd_template_pth = Path("prompts/prompt_template.txt")
with open(jd_template_pth, "r", encoding="utf-8") as f:
    jd_template_str = f.read()

jd_example_pth = Path("examples/jd_fr_1.txt")
with open(jd_example_pth, "r", encoding="utf-8") as f:
    jd_str = f.read()

jd_template = ChatPromptTemplate.from_template(jd_template_str)
output_parser = JsonOutputParser()

chain = jd_template | llm | output_parser
response = chain.invoke({"text": jd_str})
pprint(response)
