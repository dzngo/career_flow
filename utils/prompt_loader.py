from langchain.prompts import ChatPromptTemplate


def load_prompt(path):
    """
    Load a prompt template from a file path.

    Args:
        path (str): File path to the prompt text file.

    Returns:
        ChatPromptTemplate: Loaded prompt template object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return ChatPromptTemplate.from_template(f.read())
