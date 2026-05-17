"""Ultra-fast 1-2 step generation via Sana-Sprint."""
from strands import Agent
from strands_sana import sana_sprint_generate


def main():
    agent = Agent(tools=[sana_sprint_generate])
    agent("Generate a cyberpunk hummingbird in 2 steps using sana_sprint_generate")


if __name__ == "__main__":
    main()
