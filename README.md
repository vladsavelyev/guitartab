# GuitarT

Guitar tab transformer: building a model for generating guitar tablatures.

Tablature (tabs) is a common notation to represent guitar music. Unlike sheet music, tabs is easier to read for beginners as they directly show the guitar frets that need to be played. For example, the notation below is showing that the 7th fret of the 4th string needs to be played twice, followed by the 5th fret on the 3rd string, followed by an open 2nd string, and finally 4th string and the first string are played at the same time.

G|-7-7-------7-|
D|-----5-5-----|
A|---------0---|
E|-----------7-|

It's a pretty compact representation of music and, in theory should facilitate efficient music generation with transformer-based NLP models. Music is in low ways similar in text, having a structure full of self-references, full and partial repetitions, and higher-level semantics (e.g. roles of different chords in creating moods), and global composition (e.g. introduction-culmination-conclusion). That fits well to be captured by the attention mechanism. 

But first, we need to find a way to convert multi-string tabs into a flat text format, suitable for tokenization. 

### `alphaTex` format

The [alphaTex](https://alphatab.net/docs/alphatex/introduction/#:~:text=AlphaTex%20is%20a%20text%20format,the%20features%20alphaTab%20supports%20overall.) format is introduced by the [alphaTab](https://github.com/CoderLine/alphaTab) tool, open source platform that supports tools to work with tabs. We can clone the repo and install it as a node module in order to use its features to work with alphaTex.

```sh
git submodule add http://github.com/CoderLine/alphaTab.git
cd alphaTab
npm run build
cd ..
rm -rf node_modules/@coderline/alphatab/dist
cp -r alphaTab/dist node_modules/@coderline/alphatab/
```

Unfortunately, there is only a `AlphaTexImporter` modudle in the project. We digged a bit and found that the `AlphaTexExporter` module existed, but starting from [this commit](https://github.com/CoderLine/alphaTab/tree/a15680687214b4f9d85832a4152e98f4feeb5590), when the project was rewritten to TypeScript, it got removed. The last commit to have `AlphaTexExporter` was [7f82ec0aa36bbb6d7cea57785202563f677ac859](https://github.com/CoderLine/alphaTab/blob/7f82ec0aa36bbb6d7cea57785202563f677ac859/Source/AlphaTab/Exporter/AlphaTexExporter.cs). We can just use that revision instead of `main`. But we actually go a bit further down the history and use a bit earlier commit, that had pre-built JS artefacts, so we don't have to rebuild ourselves. We wget the artefact directly from the repo:

```sh
cp -r node_modules/@coderline/alphatab node_modules/@coderline/alphatab_7f82ec0a
# get the compiled JS version (last commit it was released):
wget https://github.com/CoderLine/alphaTab/raw/fd29edaf872834de50612005adcecf4a1c9597be/Build/JavaScript/AlphaTab.js \
	-O node_modules/@coderline/alphatab_7f82ec0a/dist/alphaTab.js
```

Now, to conver gp files to tex and reverse, we implemented two scripts:

```sh
node gtp_to_tex.js test/metallica.gp4 test/out-metallica.tex
node tex_to_gtp.js test/metallica.tex test/out-metallica.gp7
```

However, when tried in practice, that old `AlphaTexExporter` doesn't match the modern alphaTex format, and generates incorrect tabs. So we will attempt to re-implement it.

