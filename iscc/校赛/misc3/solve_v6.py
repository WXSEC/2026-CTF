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

def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    rem = len(bits) % 8
    if rem:
        bits = bits[:len(bits) - rem]
    bits = bits.reshape(-1, 8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()

def xor_key(data, key_bytes):
    key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    key_tiled = np.tile(key_arr, len(data_arr) // len(key_arr) + 1)[:len(data_arr)]
    return (data_arr ^ key_tiled).tobytes()

def search_flag(data, label):
    for pattern in [b'ISCC{', b'iscc{', b'flag{']:
        idx = data.find(pattern)
        if idx >= 0:
            end = data.find(b'}', idx)
            if end >= 0:
                flag = data[idx:end+1].decode('utf-8', errors='replace')
                print(f'*** FLAG FOUND [{label}]: {flag} ***')
                return True
    return False

n_rows = n // 4

print(f'n={n}, key={key}')

print('\n=== Check lag-1 autocorrelation (Manchester encoding check) ===')
lag1 = np.mean(arr[:1000000] != arr[1:1000001])
print(f'Lag-1 XOR (should be ~1.0 for Manchester): {lag1:.6f}')

print('\n=== Try Manchester decoding ===')
if lag1 > 0.9:
    print('Data looks like Manchester encoding!')
    manchester_bits = arr[::2]
    m_bytes = bits_to_bytes(manchester_bits)
    result = xor_key(m_bytes, key)
    if search_flag(result, 'Manchester+XOR'):
        sys.exit(0)
else:
    print('Not Manchester encoding')

print('\n=== Try inverse transposition with different widths ===')
for width in [2, 4, 8, 11, 16, 22, 44]:
    if n % width != 0:
        continue
    n_r = n // width
    inv = arr[:n_r*width].reshape(width, n_r).T.flatten()
    inv_bytes = bits_to_bytes(inv)
    for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
        result = xor_key(inv_bytes, key_bytes)
        if search_flag(result, f'InvTrans_w{width}+{key_name}'):
            sys.exit(0)

print('\n=== Try nibble-level inverse transposition with different widths ===')
n_nibbles = n // 4
for width in [2, 4, 8, 11, 22]:
    if n_nibbles % width != 0:
        continue
    n_r = n_nibbles // width
    nibble_data = arr[:n_nibbles*4].reshape(n_nibbles, 4)
    nibble_vals = (nibble_data * np.array([8,4,2,1])).sum(axis=1).astype(np.uint8)
    inv_nib = nibble_vals[:n_r*width].reshape(width, n_r).T.flatten()
    inv_nib_bytes = inv_nib.tobytes()
    for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
        result = xor_key(inv_nib_bytes, key_bytes)
        if search_flag(result, f'NibbleInvTrans_w{width}+{key_name}'):
            sys.exit(0)

print('\n=== Try byte-level inverse transposition with different widths ===')
raw_bytes = bits_to_bytes(arr)
raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
n_bytes = len(raw_arr)
for width in [2, 4, 8, 11, 21, 22, 44]:
    if n_bytes % width != 0:
        continue
    n_r = n_bytes // width
    inv_b = raw_arr[:n_r*width].reshape(width, n_r).T.flatten().tobytes()
    for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
        result = xor_key(inv_b, key_bytes)
        if search_flag(result, f'ByteInvTrans_w{width}+{key_name}'):
            sys.exit(0)

print('\n=== Try: Möbius strip with half XOR, different widths ===')
half = n // 2
first_half = arr[:half]
second_half = arr[half:2*half]

mobius_variants = [
    ('XOR_rev_flip', first_half ^ np.flip(1-second_half)),
    ('XOR_rev', first_half ^ np.flip(second_half)),
    ('XOR_flip', first_half ^ (1-second_half)),
    ('XOR_plain', first_half ^ second_half),
    ('first_half', first_half),
]

for mob_name, mob_data in mobius_variants:
    for width in [4, 8]:
        m_len = len(mob_data)
        if m_len % width != 0:
            continue
        m_r = m_len // width
        m_inv = mob_data[:m_r*width].reshape(width, m_r).T.flatten()
        m_bytes = bits_to_bytes(m_inv)
        for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
            result = xor_key(m_bytes, key_bytes)
            if search_flag(result, f'{mob_name}_InvTrans_w{width}+{key_name}'):
                sys.exit(0)

print('\n=== Try: XOR with key at bit level first, then inverse transposition ===')
key_bits = ''.join(bin(b)[2:].zfill(8) for b in key)
key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
key_tiled = np.tile(key_bit_arr, len(arr) // len(key_bit_arr) + 1)[:len(arr)]
xored = arr ^ key_tiled

for width in [4, 8, 11]:
    if n % width != 0:
        continue
    n_r = n // width
    xored_inv = xored[:n_r*width].reshape(width, n_r).T.flatten()
    xored_inv_bytes = bits_to_bytes(xored_inv)
    if search_flag(xored_inv_bytes, f'BitXOR_InvTrans_w{width}'):
        sys.exit(0)

print('\n=== Try: b64 key bit XOR first, then inverse transposition ===')
b64_key_bits = ''.join(bin(b)[2:].zfill(8) for b in key_b64)
b64_key_bit_arr = np.frombuffer(b64_key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
b64_key_tiled = np.tile(b64_key_bit_arr, len(arr) // len(b64_key_bit_arr) + 1)[:len(arr)]
xored_b64 = arr ^ b64_key_tiled

for width in [4, 8]:
    if n % width != 0:
        continue
    n_r = n // width
    xored_inv = xored_b64[:n_r*width].reshape(width, n_r).T.flatten()
    xored_inv_bytes = bits_to_bytes(xored_inv)
    if search_flag(xored_inv_bytes, f'B64BitXOR_InvTrans_w{width}'):
        sys.exit(0)

print('\n=== Try: RC4 decryption ===')
def rc4(data, key_bytes):
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key_bytes[i % len(key_bytes)]) % 256
        S[i], S[j] = S[j], S[i]
    i = j = 0
    result = bytearray(len(data))
    for k in range(len(data)):
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        result[k] = data[k] ^ S[(S[i] + S[j]) % 256]
    return bytes(result)

inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes(inv_bits)

for data_name, data_bytes in [('raw', raw_bytes), ('inv', inv_bytes)]:
    for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
        result = rc4(data_bytes[:500000], key_bytes)
        if search_flag(result, f'RC4_{data_name}_{key_name}'):
            sys.exit(0)

print('\n=== Try: known-plaintext attack with sliding window ===')
iscc = b'ISCC{'
iscc_arr = np.frombuffer(iscc, dtype=np.uint8)

for data_name, data_bytes in [('raw', raw_bytes), ('inv', inv_bytes)]:
    data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
        key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
        key_len = len(key_arr)
        for offset in range(key_len):
            expected = np.array([iscc_arr[j] ^ key_bytes[(offset+j) % key_len] for j in range(5)], dtype=np.uint8)
            search_len = min(len(data_arr), 5000000)
            for pos in range(0, search_len - 5):
                if data_arr[pos] == expected[0] and data_arr[pos+1] == expected[1] and data_arr[pos+2] == expected[2] and data_arr[pos+3] == expected[3] and data_arr[pos+4] == expected[4]:
                    decrypted = bytes(data_arr[pos+i] ^ key_bytes[(offset+i) % key_len] for i in range(min(100, len(data_arr)-pos)))
                    print(f'Match [{data_name}+{key_name}+off{offset}] pos={pos}: {repr(decrypted[:60])}')
                    if b'ISCC{' in decrypted:
                        end = decrypted.find(b'}', 5)
                        if end >= 0:
                            print(f'*** FLAG: {decrypted[:end+1].decode()} ***')
                            sys.exit(0)

print('\nDone - no flag found')
