import json5 as json
import json_repair
from langchain_community.adapters.openai import convert_openai_messages

from gpt_researcher.config.config import Config
from gpt_researcher.utils.llm import create_chat_completion

from loguru import logger

MODEL_MAPPINGS = {
    'gpt-4o': 'gpt-4o',
    'o3-mini': 'o3-mini-2025-01-31'
}

async def call_model(
    prompt: list,
    model: str,
    response_format: str = None,
):
    """Call the language model with better error handling and model mapping"""

    # Map custom names to valid OpenAI models
    actual_model = MODEL_MAPPINGS.get(model)
    if not actual_model:
        logger.warning(f"Invalid model '{model}', defaulting to gpt-3.5-turbo")
        actual_model = 'gpt-3.5-turbo'  # Valid fallback

    optional_params = {}
    if response_format == "json":
        optional_params = {"response_format": {"type": "json_object"}}

    cfg = Config()
    lc_messages = convert_openai_messages(prompt)

    try:
        response = await create_chat_completion(
            model=actual_model,  # Use mapped model name
            messages=lc_messages,
            temperature=0,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
        )

        if not response:
            raise ValueError("Empty response received from model")

        if response_format == "json":
            try:
                # First try to clean any markdown formatting
                cleaned_json_string = response.strip().strip('```json').strip('```')
                return json.loads(cleaned_json_string)
            except Exception as e:
                logger.warning(f"Initial JSON parsing failed: {e}, attempting repair")
                try:
                    return json_repair.loads(response)
                except Exception as e:
                    logger.error(f"JSON repair failed: {e}")
                    # Return a basic valid JSON rather than None
                    return {"error": "Failed to parse response", "raw_response": response}
        else:
            return response

    except Exception as e:
        logger.error(f"Error in calling model: {e}")
        if response_format == "json":
            # Return a valid JSON with error info rather than None
            return {
                "error": str(e),
                "title": "Error occurred",
                "sections": ["Error processing request"]
            }
        return f"Error: {str(e)}"
