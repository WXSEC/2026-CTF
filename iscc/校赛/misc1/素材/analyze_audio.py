import numpy as np
import wave
import struct

wav = wave.open(r'd:\iscc\misc1\素材\campus_broadcast.wav', 'rb')
n = wav.getnframes()
data = wav.readframes(n)
wav.close()

samples = np.array(struct.unpack(f'<{n}h', data), dtype=np.float64)
samples = samples / 32768.0

frame_rate = 44100

window = 441
envelope = []
for i in range(0, len(samples), window):
    chunk = samples[i:i+window]
    rms = np.sqrt(np.mean(chunk**2))
    envelope.append(rms)

threshold = 0.04
binary = [1 if e > threshold else 0 for e in envelope]

transitions = []
for i in range(1, len(binary)):
    if binary[i] != binary[i-1]:
        transitions.append((i * window / frame_rate, binary[i]))

on_durations = []
off_durations = []
prev_state = binary[0]
prev_time = 0
for t, state in transitions:
    duration = t - prev_time
    if prev_state == 1:
        on_durations.append(duration)
    else:
        off_durations.append(duration)
    prev_state = state
    prev_time = t

if prev_state == 1:
    on_durations.append(len(samples)/frame_rate - prev_time)

print(f'ON durations: {[f"{d:.3f}" for d in on_durations]}')
print(f'OFF durations: {[f"{d:.3f}" for d in off_durations]}')

min_on = min(on_durations)
max_on = max(on_durations)
mid_on = (min_on + max_on) / 2
print(f'ON range: {min_on:.3f} - {max_on:.3f}, mid: {mid_on:.3f}')

morse_elements = []
for d in on_durations:
    if d < mid_on:
        morse_elements.append('.')
    else:
        morse_elements.append('-')
print(f'Morse elements: {" ".join(morse_elements)}')

short_off = [d for d in off_durations if d < 0.1]
medium_off = [d for d in off_durations if 0.1 <= d < 0.5]
long_off = [d for d in off_durations if d >= 0.5]
print(f'Short OFF (<0.1s): {len(short_off)} - {[f"{d:.3f}" for d in short_off]}')
print(f'Medium OFF (0.1-0.5s): {len(medium_off)} - {[f"{d:.3f}" for d in medium_off]}')
print(f'Long OFF (>=0.5s): {len(long_off)} - {[f"{d:.3f}" for d in long_off]}')

# Build Morse string with separators
morse_str = ''
for i, d in enumerate(on_durations):
    if d < mid_on:
        morse_str += '.'
    else:
        morse_str += '-'
    if i < len(off_durations):
        off_d = off_durations[i]
        if off_d >= 0.3:
            morse_str += ' / '
        elif off_d >= 0.08:
            morse_str += ' '

print(f'Morse with separators: {morse_str}')

# Try to decode Morse
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
    '----.': '9'
}

chars = morse_str.split(' / ')
decoded = []
for char_group in chars:
    symbols = char_group.split()
    for s in symbols:
        s = s.strip()
        if s in MORSE_CODE:
            decoded.append(MORSE_CODE[s])
        elif s:
            decoded.append(f'[{s}]')

print(f'Decoded: {"".join(decoded)}')
