"""
Functions to convert from GuitarPro files to alphatex and vice versa
"""

import sys
from typing import List
import guitarpro as gp
import itertools
from pathlib import Path
import re
import asyncio
from tqdm.asyncio import tqdm_asyncio
import transliterate
import logging
import coloredlogs

fmt = "%(asctime)s %(levelname)s (%(name)s %(lineno)s): %(message)s"
coloredlogs.install(level="INFO", fmt=fmt)

__all__ = ["alphatex_to_song", "song_to_alphatex", "convert_all"]

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def song_to_alphatex(song: gp.Song) -> str:
    lines = []
    lines.append(f'\\title "{song.title}"')
    if song.subtitle:
        lines.append(f'\\subtitle "{song.subtitle}"')
    lines.append(f"\\tempo {song.tempo}")
    lines.append(".")
    for track in song.tracks:
        if not track.channel or not track.channel.instrument:
            logging.debug(
                f"Skipping track without instrument: {song.title} #{track.number} "
                f"{track.name}"
            )
            continue
        lines.append(f"\\track")
        lines.append(f"\\instrument {track.channel.instrument}")
        if track.strings:
            tuning = [
                _tuning_to_str(s.value, include_octave=True) for s in track.strings
            ]
            lines.append(f"\\tuning {' '.join(tuning)}")
        if track.fretCount:
            lines.append(f"\\frets {track.fretCount}")
        lines.append(_measures_to_str(track.measures))
    return "\n".join(lines)


def _tuning_to_str(tuning: int, include_octave=False):
    """
    Convert tuning to text representation
    :param tuning: tuning in format octave * 12 + note
    :param include_octave: whether to include octave in the result
    :return: text representation of the tuning

    >>> _tuning_to_str(49)
    'Db'
    >>> _tuning_to_str(49, include_octave=True)
    'Db4'
    """
    octave = tuning // 12
    note = tuning % 12
    result = NOTES[note]
    if include_octave:
        result += str(octave)
    return result


def _parse_tuning(tuning: str) -> int:
    """
    Parse tuning from text representation
    :param tuning: text representation of the tuning
    :return: tuning in format octave * 12 + note

    >>> _parse_tuning("Db4")
    49
    """
    octave = 4
    note = 0
    if tuning[-1].isdigit():
        octave = int(tuning[-1])
        tuning = tuning[:-1]
    note = NOTES.index(tuning)
    return octave * 12 + note


def _measures_to_str(measures: List[gp.Measure]) -> str:
    mstrs = []
    cur_dur = None
    for measure in measures:
        mstr = ""
        beats = measure.voices[0].beats
        if beats and beats[0].duration != cur_dur:
            cur_dur = beats[0].duration
            mstr += f":{cur_dur.value} "
        for beat in beats:
            if not beat.notes:
                bstr = "r"
                if beat.duration != cur_dur:
                    bstr += f".{beat.duration.value}"
                mstr += bstr + " "
                continue
            bstr = ""
            nstrs = []
            bdur = None
            for note in beat.notes:
                nstrs.append(f"{note.value}.{note.string}")
                ef = []
                if note.beat.duration.isDotted:
                    ef.append("d")
                if note.effect.vibrato:
                    ef.append("v")
                if note.effect.hammer:
                    ef.append("h")
                if note.effect.trill is not None:
                    ef.append("tr")
                if note.effect.palmMute:
                    ef.append("pm")
                if note.effect.harmonic is not None:
                    if isinstance(note.effect.harmonic, gp.NaturalHarmonic):
                        ef.append("nh")
                    elif isinstance(note.effect.harmonic, gp.ArtificialHarmonic):
                        ef.append("ah")
                    elif isinstance(note.effect.harmonic, gp.TappedHarmonic):
                        ef.append("th")
                    elif isinstance(note.effect.harmonic, gp.PinchHarmonic):
                        ef.append("ph")
                    elif isinstance(note.effect.harmonic, gp.SemiHarmonic):
                        ef.append("sh")
                if note.effect.isBend:
                    ef.append("be")
                    ef.append(
                        "("
                        + " ".join(
                            f"{p.position} {p.value}" for p in note.effect.bend.points
                        )
                        + ")"
                    )
                if note.effect.staccato:
                    ef.append("st")
                if note.effect.letRing:
                    ef.append("lr")
                if note.effect.ghostNote:
                    ef.append("g")
                if beat.effect.fadeIn:
                    ef.append("f")
                if beat.duration.tuplet.enters != 1:
                    ef.append(f"tu {beat.duration.tuplet.enters}")
                    # Finished parsing effects
                if ef:
                    nstrs[-1] += "{" + " ".join(ef) + "}"
                if beat.duration != cur_dur:
                    bdur = beat.duration.value
            if len(nstrs) > 1:
                bstr += "(" + " ".join(nstrs) + ")"
            elif len(nstrs) == 1:
                bstr = nstrs[0]
            if bdur is not None:
                bstr += f".{bdur}"
            mstr += bstr + " "
        if mstr := mstr.strip():
            mstrs.append(mstr)
    return " | \n".join(mstrs)


def _replace_spaces(s):
    """
    Replace spaces for easier parsing:
    - keep spaces separating beats,
    - replace spaces separating notes within a beat with ;
    - replace spaces separating note effects with _
    - replace spaces separating details of a note effect with -
    - remove parentheses around note effects
    """
    new_string = ""
    braces = False
    inner_parens = False
    outer_parens = False
    for char in s:
        if char == "{":
            braces = True
        elif char == "}":
            braces = False
        elif char == "(":
            if braces:
                inner_parens = True
                char = ""
            else:
                outer_parens = True
        elif char == ")":
            if braces:
                inner_parens = False
                char = ""
            else:
                outer_parens = False
        elif char == " ":
            if braces:
                char = "_"
                if inner_parens:
                    char = "-"
            elif outer_parens:
                char = ";"
        new_string += char
    return new_string


def alphatex_to_song(tex: str) -> gp.Song:
    song = gp.Song()

    def _parse_track_header(k, v):
        if k == "instrument":
            song.tracks[-1].channel.instrument = int(v)
        elif k == "tuning":
            song.tracks[-1].strings = [
                gp.GuitarString(i + 1, _parse_tuning(v))
                for i, v in enumerate(v.split(" "))
            ]
        elif k == "frets":
            song.tracks[-1].fretCount = int(v)

    lines = (l.strip() for l in tex.splitlines() if l.strip())
    for line in itertools.takewhile(lambda l: l != ".", lines):
        assert line.startswith("\\")
        line = line.lstrip("\\")
        kv = line.split(" ", 1)
        if len(kv) == 1:
            continue
        k, v = kv
        if k in [
            "title",
            "subtitle",
            "artist",
            "album",
            "words",
            "music",
            "copyright",
            "tab",
            "instructions",
        ]:
            song.__dict__[k] = v.strip('"').strip("'")
        elif k == "tempo":
            song.tempo = int(v)
        else:
            _parse_track_header(k, v)

    track_lines = []
    for line in lines:
        if line.startswith("\\"):
            line = line.lstrip("\\")
            kv = line.split(" ", 1)
            if len(kv) == 2:
                k = kv[0]
                v = kv[1].strip('"').strip("'")
            else:
                k = kv[0]
                v = None
            if k == "track":
                if track_lines:
                    _parse_track_lines(song.tracks[-1], track_lines)
                    track_lines = []
                    song.tracks.append(gp.Track(song, number=len(song.tracks) + 1))
                song.tracks[-1].name = v if v else f"Track {song.tracks[-1].number}"
            else:
                _parse_track_header(k, v)
        else:
            track_lines.append(line)
    if track_lines:
        _parse_track_lines(song.tracks[-1], track_lines)
    return song


def _parse_track_lines(track: gp.Track, lines: List[str]):
    cur_dur = gp.Duration(gp.Duration.quarter)
    for mstr in " ".join(lines).strip("|").split(" | "):
        if not (mstr := mstr.strip()):
            continue
        # Fix up so we can just split beats by space
        mstr = _replace_spaces(mstr)
        measure = track.measures[-1]
        for bstr in mstr.split(" "):
            if bstr.startswith(":"):
                cur_dur = gp.Duration(int(bstr[1:]))
                continue
            elif bstr.startswith("r"):
                beat = gp.Beat(measure.voices, duration=cur_dur)
                measure.voices[0].beats.append(beat)
                if bstr.count(".") > 0:
                    beat.duration = gp.Duration(int(bstr.rsplit(".")[-1]))
                continue
            beat = gp.Beat(measure.voices, duration=cur_dur)
            measure.voices[0].beats.append(beat)
            if "*" in bstr:
                bstr, mult = bstr.split("*")
                for _ in range(int(mult) - 1):
                    measure.voices[0].beats.append(beat)
            if (not bstr.startswith("(") and bstr.count(".") == 2) or bstr.count(
                ")."
            ) > 0:
                body, dur = bstr.rsplit(".", 1)
                beat.duration = gp.Duration(int(dur))
            else:
                body = bstr
            body = body.strip(")").strip("(")
            for nstr in body.split(";"):
                fret, string_and_ef = nstr.split(".")
                note = gp.Note(
                    beat,
                    value=int(fret),
                    string=int(string_and_ef.split("{")[0]),
                )
                beat.notes.append(note)
                if "{" in string_and_ef:
                    ef = string_and_ef.split("{")[1][:-1].strip("_").split("_")
                    _parse_effects(note, ef)
        track.measures.append(gp.Measure(track, gp.MeasureHeader()))


def _parse_effects(note, ef: List[str]):
    i = 0
    while i < len(ef):
        if ef[i] == "d":
            note.beat.duration.isDotted = True
        elif ef[i] == "v":
            note.effect.vibrato = True
        elif ef[i] == "h":
            note.effect.hammer = True
        elif ef[i] in ["b", "be"]:
            values = [int(v) for v in ef[i + 1].split("-")]
            if ef[i] == "b":
                stride = gp.BendEffect.maxPosition // (len(values) - 1)
                positions = [p * stride for p in range(len(values))]
                note.effect.bend = gp.BendEffect(
                    points=[
                        gp.BendPoint(position=p, value=v)
                        for p, v in zip(positions, values)
                    ]
                )
            else:
                note.effect.bend = gp.BendEffect(
                    points=[
                        gp.BendPoint(
                            position=values[p * 2],
                            value=values[p * 2 + 1],
                        )
                        for p in range(len(values) // 2)
                    ],
                )
            i += 1
        elif ef[i] == "g":
            note.effect.ghostNote = (True,)
        elif ef[i] == "x":
            note.type = gp.NoteType.dead
        elif ef[i] == "r":
            note.type = gp.NoteType.rest
        elif ef[i] in ["-", "t"]:
            note.type = gp.NoteType.tie
        elif ef[i] == "lr":
            note.effect.letRing = True
        elif ef[i] == "st":
            note.effect.staccato = True
        elif ef[i] == "pm":
            note.effect.palmMute = True
        elif ef[i] == "nh":
            note.effect.harmonic = gp.NaturalHarmonic()
        elif ef[i] == "ah":
            note.effect.harmonic = gp.ArtificialHarmonic()
        elif ef[i] == "th":
            note.effect.harmonic = gp.TappedHarmonic()
        elif ef[i] == "ph":
            note.effect.harmonic = gp.PinchHarmonic()
        elif ef[i] == "sh":
            note.effect.harmonic = gp.SemiHarmonic()
        elif ef[i] == "tr":
            note.effect.trill = gp.TrillEffect()
        elif ef[i] == "f":
            note.beat.effect.fadeIn = True
        elif ef[i] == "tu":
            enters = int(ef[i + 1])
            times = dict(gp.Tuplet.supportedTuplets)[enters]
            note.beat.duration.tuplet = gp.Tuplet(enters, times)
            i += 1
        i += 1


def _test():
    tex = """\
    \\title "My Song"
    \\tempo 90
    .
    \\track "First Track"
    \\instrument 42
    1.1 2.1 3.1 4.1 |
    \\track
    \\tuning A1 D2 A2 D3 G3 B3 E4
    4.1 3.1 2.1 1.1 |
    """

    song = alphatex_to_song(tex)
    tex2 = song_to_alphatex(song)
    print("Before:")
    print(tex)
    print()
    print("After:")
    print(tex2)


def _fix_path(path: Path) -> Path:
    """
    Slugify file name.
    """
    path = str(path)
    if re.search(r"[а-яА-Я]", path):
        path = transliterate.translit(path, "ru", reversed=True)
    path = re.sub(r"\s+", "_", path)  # replace whitespaces
    path = re.sub(r"\'", "-", path)  # replace apostrophes
    return Path(path)


async def convert_all():
    """
    Convert all GP files in a directory to AlphaTex.
    """
    src_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2])
    dst_path.mkdir(exist_ok=True)

    paths = list(src_path.glob("**/*.gp[3-5]"))
    logging.info(f"Found {len(paths)} GuitarPro files")

    async def _convert_one(path):
        song_name = path.stem
        song_name = re.sub(r"\(\d+\)", "", song_name).strip()

        out_path = dst_path / path.relative_to(src_path).with_suffix(".tex")
        out_path = _fix_path(out_path)
        if out_path.exists():
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            song = gp.parse(path)
        except gp.GPException as e:
            logging.warning(f"WARNING: failed to parse {path} with a GPException: {e}")
            return
        except Exception as e:
            logging.warning(f"WARNING: failed to parse {path}: {e}")
            return
        try:
            tex = song_to_alphatex(song)
        except Exception as e:
            logging.warning(f"WARNING: failed to convert {path} to alphaTex: {e}")
            return
        with out_path.open("w") as f:
            f.write(tex)

    async for path in tqdm_asyncio(paths):
        await _convert_one(path)


if __name__ == "__main__":
    asyncio.run(convert_all())
