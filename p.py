import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import sounddevice as sd
import datetime
from dataclasses import dataclass, field
import rtmidi
import time


# Helper function to convert frequency to MIDI note
def freq_to_midi(frequency):
    if frequency == 0:
        return None
    return 69 + 12 * np.log2(frequency / 440.0)


@dataclass
class Mel:
    fft_snip: np.ndarray
    fre_snip: np.ndarray

    def plot(self):
        plt.plot(self.fre_snip, np.abs(self.fft_snip))
        plt.title("FFT of Audio Data")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, 25000)
        plt.grid(True)
        plt.show()


@dataclass
class Recording:
    name: str
    rate: int = 44100
    duration: int = 10
    channels: int = 1
    dtype: str = 'int16'
    data: np.ndarray = field(default=None, repr=False)
    fft_data: np.ndarray = field(default=None, repr=False)
    freqs: np.ndarray = field(default=None, repr=False)

    def record(self):
        self.data = sd.rec(int(self.duration * self.rate), samplerate=self.rate, channels=self.channels,
                           dtype=self.dtype).flatten()
        sd.wait()

    def compute_fft(self):
        if self.data is None:
            self.record()
        self.fft_data = fft(self.data)
        self.freqs = fftfreq(len(self.fft_data), 1 / self.rate)

    def compute_mel(self, window_size=22050, speed=2):
        if self.data is None:
            self.record()
        mel = []
        step = int(window_size / speed)
        for start in range(0, len(self.data) - window_size + 1, step):
            end = start + window_size
            fft_snip = fft(self.data[start:end])
            freq_snip = fftfreq(len(fft_snip), 1 / self.rate)
            mel.append(Mel(fft_snip, freq_snip))
        return mel

    def plot_freq(self):
        if self.freqs is None or self.fft_data is None:
            self.compute_fft()
        plt.plot(self.freqs[:len(self.freqs) // 2], np.abs(self.fft_data)[:len(self.fft_data) // 2])
        plt.title("FFT of Audio Data")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, 25000)
        plt.grid(True)
        plt.show()


class MIDIEvent:

    def __init__(self, status: int, note_number: int, velocity: int):
        self.status = int(status)
        self.note_number = int(note_number)
        self.velocity = int(velocity)

    def send(self, midiout):
        """Send MIDI event through the given MIDI output, handling timing."""
        with midiout:
            if self.note_number > 127:
                self.note_number = 127
            midiout.send_message([self.status, self.note_number, 50])

    @classmethod
    def from_freq(cls, frequency, velocity=50):
        midi_note = freq_to_midi(frequency)
        if midi_note is None:
            return None
        return cls(status=144, note_number=midi_note, velocity=velocity)


@dataclass
class AudioProcessor:
    recordings: dict = field(default_factory=dict)

    def record_audio(self, name: str, duration: int):
        recording = Recording(name=name, duration=duration)
        recording.record()
        recording.compute_fft()
        self.recordings[name] = recording

    def generate_midi_events(self, name: str):
        recording = self.recordings.get(name)
        if not recording:
            raise ValueError("Recording not found!")
        mels = recording.compute_mel()
        midi_events = [MIDIEvent.from_freq(np.abs(mel.fft_snip).max(), 50) for mel in mels if
                       MIDIEvent.from_freq(np.abs(mel.fft_snip).max(), 50) is not None]
        return midi_events

    def send_midi(self, name: str, midiout):
        """Send MIDI events for a recording to a given MIDI output."""
        midi_events = self.generate_midi_events(name)
        for event in midi_events:
            if event:
                event.send(midiout)
                time.sleep(0.5)


# Example usage
processor = AudioProcessor()
processor.record_audio("test_1", 2)
recording = processor.recordings["test_1"]
recording.plot_freq()

# Setup MIDI output
midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
if available_ports:
    print("Available MIDI ports:", available_ports)
    midiout.open_port(0)  # Change the index based on your setup

# Send MIDI events
processor.send_midi("test_1", midiout)

# Cleanup
del midiout
