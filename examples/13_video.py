"""Generate a 5-second video clip with SANA-Video."""
from strands import Agent
from strands_sana import sana_video_generate


def main():
    agent = Agent(tools=[sana_video_generate])
    agent(
        "Generate a 5-second video of a cyberpunk cityscape with neon rain, "
        "using sana-video-2b-480"
    )


if __name__ == "__main__":
    main()
