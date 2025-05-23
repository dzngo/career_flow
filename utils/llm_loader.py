def get_llm(model_name: str):
    """
    Return an LLM instance based on the model name.

    Args:
        model_name (str): Name of the LLM to use.

    Returns:
        BaseLanguageModel: An instance of the selected LLM.
    """
    if model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"]:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name)

    raise ValueError(f"Unsupported LLM model: {model_name}")
