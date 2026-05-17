"""Basic Sana generation via Strands Agent."""
from strands import Agent
from strands_sana import sana_generate


def main():
    agent = Agent(tools=[sana_generate])
    agent("Generate a serene Japanese garden with cherry blossoms at sunrise, photorealistic")


if __name__ == "__main__":
    main()
