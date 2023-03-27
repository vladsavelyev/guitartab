from gp2tex import alphatex_to_song, track_to_alphatex


def test_canonrock():
    """
    Full round of conversion from .txt->.gp->.tex->.gp and compare
    """
    tex = """
    \\title "Canon Rock"
    \\subtitle "JerryC"
    \\tempo 90
    .
    :2 19.2{v f} 17.2{v f} |
    15.2{v f} 14.2{v f} |
    12.2{v f} 10.2{v f} |
    12.2{v f} 14.2{v f}.4 :8 15.2 17.2 |
    14.1.2 :8 17.2 15.1 14.1{h} 17.2 |
    15.2{v d}.4 :16 17.2{h} 15.2 :8 14.2 14.1 17.1{b (0 4 4 0)}.4 |
    15.1.8 :16 14.1{tu 3} 15.1{tu 3} 14.1{tu 3} :8 17.2 15.1 14.1 :16 12.1{tu 3} 14.1{tu 3} 12.1{tu 3} :8 15.2 14.2 |
    12.2 14.3 12.3 15.2 :32 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h} 14.2{h} 15.2{h}"""

    song = alphatex_to_song(tex)
    # gp.write(song, "test/results/canon_rock.gp")

    tex2 = track_to_alphatex(song.tracks[0])

    song2 = alphatex_to_song(tex2)
    # gp.write(song2, "test/results/canon_rock2.gp")

    assert len(song2.measureHeaders) == len(song.measureHeaders)
    assert len(song2.tracks[0].measures) == len(song.tracks[0].measures)
    assert song2.tracks[0] == song.tracks[0]
