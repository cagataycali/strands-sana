"""Batch generation example."""
from strands import Agent
from strands_sana import sana_batch


def main():
    agent = Agent(tools=[sana_batch])
    agent(
        "Generate 3 images: a cat astronaut, a dog pirate, a duck cyborg. Use sana_batch."
    )


if __name__ == "__main__":
    main()
