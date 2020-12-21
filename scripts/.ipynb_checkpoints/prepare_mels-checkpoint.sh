#!/usr/bin/env bash

set -e

DATADIR="korean_dataset"
FILELISTSDIR="filelists"

TESTLIST="$FILELISTSDIR/kss_test.txt"
TRAINLIST="$FILELISTSDIR/kss_train.txt"
VALLIST="$FILELISTSDIR/kss_val.txt"

TESTLIST_MEL="$FILELISTSDIR/mel_kss_test.txt"
TRAINLIST_MEL="$FILELISTSDIR/mel_kss_train.txt"
VALLIST_MEL="$FILELISTSDIR/mel_kss_val.txt"

mkdir -p "$DATADIR/mels"
if [ $(ls $DATADIR/mels | wc -l) -ne 13100 ]; then
    python preprocess_audio2mel.py --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
    python preprocess_audio2mel.py --wav-files "$TESTLIST" --mel-files "$TESTLIST_MEL"
    python preprocess_audio2mel.py --wav-files "$VALLIST" --mel-files "$VALLIST_MEL"	
fi	
