"""Use the pipeline directly without an Agent."""
from strands_sana.pipeline import get_pipeline
from strands_sana.utils import save_image


def main():
    pipe = get_pipeline("sana-0.6b-512")
    images = pipe.generate(
        prompt="a peaceful mountain lake at dawn, oil painting",
        num_inference_steps=20,
        seed=42,
    )
    path = save_image(images[0])
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
