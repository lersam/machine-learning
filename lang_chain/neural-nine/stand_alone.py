import logging
import httpx

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = init_chat_model(
    model="ollama:gemma3:4b-it-qat",
    temperature=0.1,
)

def get_model_response(conversation):
    try:
        response = model.invoke(conversation)
        logger.info("----------------")
        logger.info(response)
        logger.info("----------------")
        logger.info("content %s", response.content) 
        return response
    except httpx.HTTPError as e:
        logger.error("HTTP error occurred: %s", e)
    except Exception as e:
        logger.error("An error occurred: %s", e)

def get_model_streaming_response(conversation):
    try:
        for chunk in model.stream(conversation):
            print(chunk.text, end="", flush=True)
    except httpx.HTTPError as e:
        logger.error("HTTP error occurred: %s", e)
    except Exception as e:
        logger.error("An error occurred: %s", e)

if __name__ == "__main__":
    conversation = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, what is Python?"),
        AIMessage(content="Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used for web development, data analysis, artificial intelligence, scientific computing, and more."),
        HumanMessage(content="When was it released?"),
    ]

    get_model_streaming_response(conversation)
