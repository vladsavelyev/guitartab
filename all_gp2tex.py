from pathlib import Path
import guitarpro as gp
from tqdm.asyncio import tqdm_asyncio
import re
from gp2tex import song_to_alphatex


src_path = Path("/Users/vlad/ArchivedGoogleDrive/PlayMusic/tabs")
dst_path = Path("/Users/vlad/ArchivedGoogleDrive/PlayMusic/tabs_tex")
dst_path.mkdir(exist_ok=True)

paths = list(src_path.glob("**/*.gp[3-5]"))
print(f"Found {len(paths)} GuitarPro files")


import asyncio
from multiprocessing import Pool


async def convert_file(path: Path):
    song_name = path.stem
    song_name = re.sub(r"\(\d+\)", "", song_name).strip()

    out_path = dst_path / path.relative_to(src_path).with_suffix(".tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    try:
        song = gp.parse(path)
    except gp.GPException as e:
        print(f"WARNING: failed to parse {path} with a GPException: {e}")
        return
    except Exception as e:
        print(f"WARNING: failed to parse {path}: {e}")
        return
    try:
        tex = song_to_alphatex(song)
    except Exception as e:
        print(f"WARNING: failed to convert {path} to alphaTex: {e}")
        return
    with out_path.open("w") as f:
        f.write(tex)


async def main():
    async for path in tqdm_asyncio(paths):
        await convert_file(path)


if __name__ == "__main__":
    asyncio.run(main())
