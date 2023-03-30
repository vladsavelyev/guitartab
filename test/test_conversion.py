from gp2tex import alphatex_to_song, track_to_alphatex


def _test_conversion(tex):
    """
    Full round of conversion from .txt->.gp->.tex->.gp and compare
    """
    song = alphatex_to_song(tex)
    tex2 = track_to_alphatex(song.tracks[0])
    song2 = alphatex_to_song(tex2)

    assert len(song2.measureHeaders) == len(song.measureHeaders)
    assert len(song2.tracks[0].measures) == len(song.tracks[0].measures)
    assert song2.tracks[0] == song.tracks[0]

    tex3 = track_to_alphatex(song2.tracks[0])
    assert tex2 == tex3


def test_canonrock():
    _test_conversion(
        """
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
    )


def test_chords():
    _test_conversion(
        """
\\title "Some chords"
.
(0.3{v h} 0.4{b (0 4)}).4 (3.3 3.4).4 (5.3 5.4).4 r.8 (0.3 0.4).8 |
r.8 (3.3 3.4).8 r.8 (6.3 6.4).8 (5.3 5.4).4 r.4 |
(0.3 0.4).4 (3.3 3.4).4 (5.3 5.4).4 r.8 (3.3 3.4).8 |
r.8 (0.3 0.4).8
"""
    )


def test_repeated():
    _test_conversion(
        """
\\title "Repeated notes"
.
3.3*4 | 4.3*4
"""
    )


def test_multitrack():
    _test_conversion(
        """
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
    )
