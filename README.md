# mic2midi

Thid repo:
1. Extracts raw microphone data.
2. Extracts sliding window frequencies from data by applying FFT.
3. Extracts (note, velocity) data from the frequency data, to create list of MIDI events.
4. Plays those MIDI events on some output synthesyzer or plugged in instrument, such as a piano.
