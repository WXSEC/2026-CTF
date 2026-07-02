import base64
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

key_b64 = b'WXRoOVVyMDYyYXpaQTA5eTRyczVM'
key = base64.b64decode(key_b64)

with open('truth.dat', 'r') as f:
    truth = f.read().strip()

n = len(truth)
arr = np.frombuffer(truth.encode('ascii'), dtype=np.uint8) - ord('0')
del truth

n_rows = n // 4

def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    rem = len(bits) % 8
    if rem:
        bits = bits[:len(bits) - rem]
    bits = bits.reshape(-1, 8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()

def xor_bytes(data, key_bytes):
    key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    key_tiled = np.tile(key_arr, len(data_arr) // len(key_arr) + 1)[:len(data_arr)]
    return (data_arr ^ key_tiled).tobytes()

def xor_bits(bits_arr, key_bytes):
    key_bits = ''.join(bin(b)[2:].zfill(8) for b in key_bytes)
    key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
    key_tiled = np.tile(key_bit_arr, len(bits_arr) // len(key_bit_arr) + 1)[:len(bits_arr)]
    return bits_arr ^ key_tiled

def search_flag(data, label):
    text = data.decode('utf-8', errors='replace')
    for pattern in ['ISCC{', 'ISCC', 'iscc{', 'iscc', 'flag{', 'Flag{']:
        idx = text.find(pattern)
        if idx >= 0:
            end = text.find('}', idx)
            if end >= 0:
                print(f'*** FLAG FOUND [{label}]: {text[idx:end+1]} ***')
                return True
            print(f'[{label}] {pattern} at {idx}: {repr(text[idx:idx+80])}')
    hx = data.hex()
    for pattern in ['49534343', '69736363']:
        idx = hx.find(pattern)
        if idx >= 0:
            print(f'[{label}] hex match at byte {idx//2}: {data[idx//2:idx//2+30]}')
            return True
    return False

def check_readable(data, label, threshold=0.3):
    printable = sum(1 for b in data[:1000] if 32 <= b <= 126)
    ratio = printable / min(len(data), 1000)
    if ratio > threshold:
        text = data[:200].decode('utf-8', errors='replace')
        print(f'[{label}] Readable ratio: {ratio:.2f}, first 100: {repr(text[:100])}')
        return True
    return False

print('=== Approach A: Nibble reversal (reverse bits within each 4-bit group) ===')
nibbles = arr[:n_rows*4].reshape(-1, 4)
nibbles_rev = np.flip(nibbles, axis=1)
rev_bits = nibbles_rev.flatten()

inv_rev = rev_bits.reshape(4, n_rows).T.flatten()
inv_rev_bytes = bits_to_bytes(inv_rev)
result = xor_bytes(inv_rev_bytes, key)
if search_flag(result, 'NibbleRev_InvTrans+XOR'):
    sys.exit(0)
check_readable(result, 'NibbleRev_InvTrans+XOR')

rev_direct = bits_to_bytes(rev_bits)
result = xor_bytes(rev_direct, key)
if search_flag(result, 'NibbleRev_Direct+XOR'):
    sys.exit(0)

print('\n=== Approach B: Byte nibble swap (swap high/low nibbles) ===')
raw_bytes = bits_to_bytes(arr)
raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
swapped = ((raw_arr & 0x0F) << 4) | ((raw_arr & 0xF0) >> 4)
result = xor_bytes(swapped.tobytes(), key)
if search_flag(result, 'NibbleSwap+XOR'):
    sys.exit(0)
check_readable(result, 'NibbleSwap+XOR')

print('\n=== Approach C: Gray code to binary conversion ===')
gray_bits = arr.copy()
binary_bits = gray_bits.copy()
for i in range(1, len(binary_bits)):
    binary_bits[i] = gray_bits[i] ^ binary_bits[i-1]

inv_gray = binary_bits[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_gray_bytes = bits_to_bytes(inv_gray)
result = xor_bytes(inv_gray_bytes, key)
if search_flag(result, 'GrayToBin_InvTrans+XOR'):
    sys.exit(0)
check_readable(result, 'GrayToBin_InvTrans+XOR')

gray_direct = bits_to_bytes(binary_bits)
result = xor_bytes(gray_direct, key)
if search_flag(result, 'GrayToBin_Direct+XOR'):
    sys.exit(0)

print('\n=== Approach D: Different transposition widths ===')
for width in [8, 16, 32]:
    if n % width != 0:
        continue
    w_rows = n // width
    inv_w = arr[:w_rows*width].reshape(width, w_rows).T.flatten()
    inv_w_bytes = bits_to_bytes(inv_w)
    result = xor_bytes(inv_w_bytes, key)
    label = f'InvTrans_w{width}+XOR'
    if search_flag(result, label):
        sys.exit(0)
    check_readable(result, label)

print('\n=== Approach E: Bit rotation within nibbles ===')
for rot in [1, 2, 3]:
    nibbles = arr[:n_rows*4].reshape(-1, 4)
    rotated = np.roll(nibbles, rot, axis=1)
    rot_bits = rotated.flatten()
    inv_rot = rot_bits.reshape(4, n_rows).T.flatten()
    inv_rot_bytes = bits_to_bytes(inv_rot)
    result = xor_bytes(inv_rot_bytes, key)
    label = f'Rot{rot}_InvTrans+XOR'
    if search_flag(result, label):
        sys.exit(0)
    check_readable(result, label)

print('\n=== Approach F: XOR with key at different offsets ===')
inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes(inv_bits)
inv_arr = np.frombuffer(inv_bytes, dtype=np.uint8)

for offset in range(len(key)):
    key_shifted = np.roll(np.frombuffer(key, dtype=np.uint8), offset)
    key_tiled = np.tile(key_shifted, len(inv_arr) // len(key_shifted) + 1)[:len(inv_arr)]
    result = (inv_arr ^ key_tiled).tobytes()
    text = result[:200].decode('utf-8', errors='replace')
    if 'ISCC' in text:
        print(f'*** FOUND at offset {offset}: {repr(text[:100])} ***')
        search_flag(result, f'Offset_{offset}')
        sys.exit(0)

print('\n=== Approach G: RC4 with full data (first 1MB) ===')
def rc4(data, key_bytes):
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key_bytes[i % len(key_bytes)]) % 256
        S[i], S[j] = S[j], S[i]
    i = j = 0
    result = bytearray()
    for byte in data:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        result.append(byte ^ S[(S[i] + S[j]) % 256])
    return bytes(result)

for data, name in [(raw_bytes[:102400], 'Direct'), (inv_bytes[:102400], 'InvTrans')]:
    for k, kn in [(key, 'key'), (key_b64, 'b64')]:
        result = rc4(data, k)
        if search_flag(result, f'RC4_{name}_{kn}'):
            sys.exit(0)
        check_readable(result, f'RC4_{name}_{kn}')

print('\n=== Approach H: Mobius at column level - check column correlations ===')
seg = n_rows
segs = [arr[i*seg:(i+1)*seg] for i in range(4)]

for i in range(4):
    for j in range(i+1, 4):
        for flip in [False, True]:
            for rev in [False, True]:
                s2 = segs[j].copy()
                if flip: s2 = 1 - s2
                if rev: s2 = np.flip(s2)
                match = np.mean(segs[i] == s2)
                if match > 0.55:
                    print(f'Segs {i} vs {j}: flip={flip}, rev={rev}, match={match:.4f}')

print('\n=== Approach I: Try reading truth.dat as hex-encoded data ===')
try:
    hex_data = bytes.fromhex(truth[:10000])
    print(f'Hex decode first 20 bytes: {hex_data[:20]}')
except:
    print('Not valid hex data')

print('\n=== Approach J: Check if data has periodic structure ===')
for period in [8, 16, 21, 32, 168]:
    matches = 0
    total = 0
    for i in range(period, min(n, 100000)):
        if arr[i] == arr[i - period]:
            matches += 1
        total += 1
    ratio = matches / total if total > 0 else 0
    print(f'Period {period}: auto-correlation = {ratio:.4f}')

print('\n=== Approach K: Mobius strip with XOR - check if XOR reveals structure ===')
xored = xor_bits(arr, key)
half = n // 2
xfirst = xored[:half]
xsecond = xored[half:2*half]

for flip in [False, True]:
    for rev in [False, True]:
        s2 = xsecond.copy()
        if flip: s2 = 1 - s2
        if rev: s2 = np.flip(s2)
        match = np.mean(xfirst == s2)
        if match > 0.55:
            print(f'XOR Mobius: flip={flip}, rev={rev}, match={match:.4f}')

print('\nDone with creative approaches.')
