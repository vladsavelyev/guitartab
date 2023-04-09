from pathlib import Path

import guitarpro as gp
from transformers import (
    pipeline,
)

from alphatex import alphatex_to_song
from guitartab import load_model, load_tokenizer, load_generation_config


def generate_song(
    out_dir: Path,
    title: str,
    device: str,
    model=None,
    tokenizer=None,
    **kwargs,
):
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=load_generation_config(),
        **kwargs,
    )

    for i, result in enumerate(generator([r"\title"])[0]):
        tex = result["generated_text"]
        tex = tex.replace(tokenizer.eos_token, "")
        tex = tex.rsplit("|", 1)[0]
        print("-" * 80)
        print(tex)
        print("-" * 80)
        try:
            song = alphatex_to_song(tex)
        except Exception as e:
            print("Could not parse the tex to GP:", e)
        else:
            try:
                fname = "".join(c if c.isalnum() else "_" for c in title).rstrip("_")
                path = out_dir / f"{fname}_{i}.gp"
                gp.write(song, str(path))
            except Exception as e:
                print("Could not write the GP file:", e)
            else:
                print("Saved song to", path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(".").resolve())
    parser.add_argument("--title", type=str, default="Untitled")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    generate_song(args.out_dir, args.title, args.device)
