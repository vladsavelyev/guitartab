"""
\title "Canon Rock"
\subtitle "JerryC"
\tempo 90
.
:2 19.2{v f} 17.2{v f} |
15.2{v f} 14.2{v f} |
12.2{v f} 10.2{v f} |
12.2{v f} 14.2{v f}.4 :8 15.2 17.2 |
14.1.2 :8 17.2 15.1 14.1{h} 17.2 |
15.2{v d}.4 :16 17.2{h} 15.2 :8 14.2 14.1 17.1{b (0 4 4 0)}.4 |
15.1.8 :16 14.1{tu 3} 15.1{tu 3} 14.1{tu 3} :8 17.2 15.1 14.1 :16 12.1{tu 3} 14.1{tu 3} 12.1{tu 3} :8 15.2 14.2 |
12.2 14.3 12.3 15.2 :32 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h}
"""

import guitarpro as gp
import re
from pathlib import Path
import itertools


def find_bass_track(song: gp.Song) -> gp.Track | None:
    N_FRETS = 24
    N_STRINGS = 4  # base; 6 for standard guitar
    INSTRUMENTS = range(32, 40)  # range(24, 31) for standard guitar

    for track in song.tracks:
        if all(
            [
                track.settings.tablature,
                len(track.strings) == N_STRINGS,
                track.fretCount == N_FRETS,
                track.channel.instrument in INSTRUMENTS,
            ]
        ):
            return track
    return None


def track_to_alphatex(track: gp.Track) -> str:
    header = []
    if hasattr(track.song, "title"):
        header.append(f'\\title "{track.song.title} ({track.name})"')
    if hasattr(track.song, "subtitle"):
        header.append(f'\\subtitle "{track.song.subtitle}"')
    if hasattr(track.song, "tempo"):
        header.append(f"\\tempo {track.song.tempo}")
    header.append(".")

    mstrs = []
    curduration = None
    for measure in track.measures:
        mstr = ""
        beats = measure.voices[0].beats
        if beats and beats[0].duration != curduration:
            curduration = beats[0].duration
            mstr += f":{curduration.value} "
        for beat in beats:
            note = beat.notes[0]
            mstr += f"{note.value}.{note.string}"
            effects = []
            if note.beat.duration.isDotted:
                effects.append("d")
            if note.effect.vibrato:
                effects.append("v")
            if note.effect.hammer:
                effects.append("h")
            if note.effect.trill is not None:
                effects.append("tr")
            if note.effect.palmMute:
                effects.append("pm")
            if note.effect.harmonic is not None:
                if isinstance(note.effect.harmonic, gp.NaturalHarmonic):
                    effects.append("nh")
                elif isinstance(note.effect.harmonic, gp.ArtificialHarmonic):
                    effects.append("ah")
                elif isinstance(note.effect.harmonic, gp.TappedHarmonic):
                    effects.append("th")
                elif isinstance(note.effect.harmonic, gp.PinchHarmonic):
                    effects.append("ph")
                elif isinstance(note.effect.harmonic, gp.SemiHarmonic):
                    effects.append("sh")
            if note.effect.isBend:
                effects.append("be")
                effects.append(
                    "("
                    + " ".join(
                        f"{p.position} {p.value}" for p in note.effect.bend.points
                    )
                    + ")"
                )
            if note.effect.staccato:
                effects.append("st")
            if note.effect.letRing:
                effects.append("lr")
            if note.effect.ghostNote:
                effects.append("g")
            if note.beat.effect.fadeIn:
                effects.append("f")
            if note.beat.duration.tuplet.enters != 1:
                effects.append(f"tu {note.beat.duration.tuplet.enters}")
            if effects:
                mstr += "{" + " ".join(effects) + "}"
            if note.beat.duration != curduration:
                mstr += f".{note.beat.duration.value}"
            mstr += " "
        if mstr := mstr.strip():
            mstrs.append(mstr)
    return "\n".join(header) + "\n" + " | \n".join(mstrs)


def alphatex_to_song(tex: str) -> gp.Song:
    lines = (l.strip() for l in tex.splitlines())

    song = gp.Song()

    for line in itertools.takewhile(lambda l: l != ".", lines):
        if line.startswith("\\"):
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
                song.__dict__[k] = v.strip('"')
            elif k == "tempo":
                song.tempo = int(v)

    track = song.tracks[0]
    curduration = gp.Duration(gp.Duration.quarter)
    for mstr in " ".join(lines).split(" | "):
        # fix up so we can just split beats by space.
        # replace space inside parens with dash:
        mstr = re.sub(
            r"\((.*?)\)", lambda match: match.group(1).replace(" ", "-"), mstr
        )
        # replace other spaces inside braces with underscore:
        mstr = re.sub(r"{(.*?)}", lambda match: match.group(0).replace(" ", "_"), mstr)
        measure = track.measures[-1]
        tokens = mstr.split(" ")
        for token in tokens:
            if token.startswith(":"):
                curduration = gp.Duration(int(token[1:]))
                continue
            beat = gp.Beat(measure.voices, duration=curduration)
            measure.voices[0].beats.append(beat)
            parts = token.split(".")
            note = gp.Note(
                beat,
                value=int(parts[0]),
                string=int(parts[1].split("{")[0]),
            )
            beat.notes.append(note)
            if len(parts) == 3:
                beat.duration = gp.Duration(int(parts[2]))
            if "{" in parts[1]:
                ef = parts[1].split("{")[1][:-1].strip("_").split("_")
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
                                        position=values[p * 2], value=values[p * 2 + 1]
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
        song.newMeasure()
    return song


# song = gp.parse("test/metallica.gp4")
# bass = find_bass_track(song)
# if not bass:
#     raise ValueError("No bass track found")
# else:
#     print(f"Found bass track: {bass.name} (number {bass.number})")
# print(track_to_alphatex(bass))


with Path("test/canon_rock.tex").open() as f:
    tex = f.read()
song = alphatex_to_song(tex)
gp.write(song, "test/results/canon_rock.gp4")

tex2 = track_to_alphatex(song.tracks[0])
with Path("test/results/canon_rock.tex").open("w") as f:
    f.write(tex2)

song2 = alphatex_to_song(tex2)
assert song2
gp.write(song2, "test/results/canon_rock2.gp4")
