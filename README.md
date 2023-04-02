# GuitarTab

Training transformer for understanding guitar tablatures in alphaTex format.

Tablature (tabs) is a common notation to represent guitar music. Unlike sheet music, tabs is easier to read for beginners as they directly show the guitar frets that need to be played. For example, the notation below is showing that the 7th fret of the 4th string needs to be played twice, followed by the 5th fret on the 3rd string, followed by an open 2nd string, and finally 4th string and the first string are played at the same time.

G|-7-7-------7-|
D|-----5-5-----|
A|---------0---|
E|-----------7-|

It's a pretty compact representation of music and, in theory should facilitate efficient music generation with transformer-based NLP models. Music is in low ways similar in text, having a structure full of self-references, full and partial repetitions, and higher-level semantics (e.g. roles of different chords in creating moods), and global composition (e.g. introduction-culmination-conclusion). That fits well to be captured by the attention mechanism. 

But first, we need to find a way to convert multi-string tabs into a flat text format, suitable for tokenization. 

### `alphaTex` format

The [alphaTex](https://alphatab.net/docs/alphatex/introduction) format is introduced by the [alphaTab](https://github.com/CoderLine/alphaTab) tool, an open source platform that supports tools to work with tabs. Unfortunately, to operate with alphaTex, the project provides only `AlphaTexImporter`. Digging a bit further, one can find that `AlphaTexExporter` module existed before, but was removed after [this commit](https://github.com/CoderLine/alphaTab/tree/a15680687214b4f9d85832a4152e98f4feeb5590) - when the project was rewritten to TypeScript. The last commit to have `AlphaTexExporter` is [7f82ec0aa36bbb6d7cea57785202563f677ac859](https://github.com/CoderLine/alphaTab/blob/7f82ec0aa36bbb6d7cea57785202563f677ac859/Source/AlphaTab/Exporter/AlphaTexExporter.cs). However, it's incompatible with the most recent `AlphaTexImporter`.

Instead of forking the TypeScript/JavaScript project, we desided to go another root: we used the [PyGuitarPro](https://github.com/Perlence/PyGuitarPro) Python port of alphaTab whithatch can parse GP files, and implemented our own [exporter](gp_to_tex.py) from PyGuitarPro abstractions into alphaTex.

We used the exporter to convert available 49014 gp3, gp4 and gp5 files from a tab archive, resulting in 47299 alphaTex files (some failed parsing), or 188470 tracks, which we contributed to [HuggingFace](https://huggingface.co/datasets/vldsavelyev/guitar_tab). Then we trained a GPT2-like model using [train.py](train.py) on a single GPU, on 3 epochs, and contributed it as [guitar_tab_gpt2](https://huggingface.co/vldsavelyev/guitar_tab_gpt2), along with a [bass-only version](https://huggingface.co/vldsavelyev/guitartab_bass).