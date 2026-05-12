import logging
import httpx

from langchain.agents import create_agent
from langchain.tools import tool

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    logger.info(f"Getting weather for {city}")
    response = httpx.get(f"https://wttr.in/{city}?format=j1")
    if response.status_code != httpx.codes.OK:
        logger.error(f"Failed to get weather data: {response.text}")
        return "Sorry, I couldn't get the weather data right now."

    return response.json()

def main() -> None:
    """Run the lang_chain module entrypoint."""
    agent = create_agent(
        model="ollama:qwen3.5:2b",
        tools=[get_weather],
        # system_prompt="You are a helpful weather assistant, who allways cracks jokes and is humorous, while remaining helpful and informative.",
        system_prompt="You are a helpful weather assistant.",
    )
    city = "Jerusalem"
    response = agent.invoke(
        {"messages": [{"role": "user", "content": f"What's the weather in {city}?"}]}
    )
    logger.info("Weather in city: %S is %s", city, response["messages"][-1].content)


if __name__ == "__main__":
    main()
