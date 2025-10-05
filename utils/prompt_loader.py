def load_prompt(path):
    """
    Load a prompt template from a file path.

    Args:
        path (str): File path to the prompt text file.

    Returns:
        str: Loaded prompt template text.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
