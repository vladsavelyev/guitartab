import guitarpro as gp
import itertools


BASS_STRINGS = [gp.GuitarString(i, s) for i, s in enumerate([43, 38, 33, 28])]


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
    header.append(f'\\title "{track.song.title}"')
    if track.song.subtitle:
        header.append(f'\\subtitle "{track.name}"')
    header.append(f"\\tempo {track.song.tempo}")
    if track.channel and track.channel.instrument:
        header.append(f"\\instument {track.channel.instrument}")
    if track.strings:
        header.append(f"\\strings {len(track.strings)}")
    if track.fretCount:
        header.append(f"\\frets {track.fretCount}")
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
            # TODO: handle pauses
            if not beat.notes:
                bstr = "r"
                if beat.duration != curduration:
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
                if beat.duration != curduration:
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
    return "\n".join(header) + "\n" + " | \n".join(mstrs)


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
    track = song.tracks[0]
    lines = (l.strip() for l in tex.splitlines())
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
            elif k == "instrument":
                track.channel.instrument = int(v)
            elif k == "strings":
                if int(v) == 4:
                    track.strings = BASS_STRINGS
            elif k == "frets":
                track.fretCount = int(v)

    curduration = gp.Duration(gp.Duration.quarter)
    for mstr in " ".join(lines).split(" | "):
        if not (mstr := mstr.strip()):
            continue
        # Fix up so we can just split beats by space
        mstr = _replace_spaces(mstr)
        measure = track.measures[-1]
        for bstr in mstr.split(" "):
            if bstr.startswith(":"):
                curduration = gp.Duration(int(bstr[1:]))
                continue
            elif bstr.startswith("r"):
                beat = gp.Beat(measure.voices, duration=curduration)
                measure.voices[0].beats.append(beat)
                if bstr.count(".") > 0:
                    beat.duration = gp.Duration(int(bstr.rsplit(".")[-1]))
                continue
            beat = gp.Beat(measure.voices, duration=curduration)
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
        song.newMeasure()
    return song


# song = gp.parse("test/data/metallica.gp4")
# bass = find_bass_track(song)
# if not bass:
#     raise ValueError("No bass track found")
# else:
#     print(f"Found bass track: {bass.name} (number {bass.number})")
# tex = track_to_alphatex(bass)
# print(tex)
# song2 = alphatex_to_song(tex)
# gp.write(song2, "test/results/metallica2.gp4")


# tex = """
# \\title "Some chords"
# .
# (0.3{v h} 0.4{b (0 4)}).4 (3.3 3.4).4 (5.3 5.4).4 r.8 (0.3 0.4).8 |
# r.8 (3.3 3.4).8 r.8 (6.3 6.4).8 (5.3 5.4).4 r.4 |
# (0.3 0.4).4 (3.3 3.4).4 (5.3 5.4).4 r.8 (3.3 3.4).8 |
# r.8 (0.3 0.4).8
# """

# # tex = """
# # \\title "Repeated notes"
# # .
# # 3.3*4 | 4.3*4
# # """

# song = alphatex_to_song(tex)
# tex2 = track_to_alphatex(song.tracks[0])
# print("Before:")
# print(tex)
# print()
# print("After:")
# print(tex2)
# song2 = alphatex_to_song(tex2)
# tex3 = track_to_alphatex(song2.tracks[0])
# print()
# print("Second round:")
# print(tex3)
