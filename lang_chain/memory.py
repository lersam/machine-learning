import logging
import httpx

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool("locate_user", description="look up the user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    match runtime.context.user_id:
        case "user_123":
            return "Jerusalem"
        case "user_456":
            return "New York"
        case _:
            return "Unknown"


@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    logger.info("Getting weather for %s", city)
    response = httpx.get(f"https://wttr.in/{city}?format=j1")
    if response.status_code != httpx.codes.OK:
        logger.error("Failed to get weather data: %s", response.text)
        return "Sorry, I couldn't get the weather data right now."

    return response.json()


def main() -> None:
    """Run the lang_chain module entrypoint."""
    model = init_chat_model("ollama:qwen3:14b", temperature=0.15)

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_weather, locate_user],
        system_prompt="You are a helpful weather assistant, who allways cracks jokes and is humorous, while remaining helpful and informative.",
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": 1}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather like?"}]},
        config=config,
        context=Context(user_id="user_123"),
    )

    structured = response.get("structured_response")
    if structured is not None:
        logger.info("Structured response: %s", structured)
        return

    messages = response.get("messages", [])
    if messages:
        last_message = messages[-1]
        content: Any = getattr(last_message, "content", last_message)
        logger.info("Agent response: %s", content)
        return

    logger.info("Agent response payload: %s", response)


if __name__ == "__main__":
    main()
