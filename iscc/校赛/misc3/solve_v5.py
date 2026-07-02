import base64
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

key_b64 = b'WXRoOVVyMDYyYXpaQTA5eTRyczVM'
key = base64.b64decode(key_b64)
print(f'Key: {key}, len: {len(key)}')

with open('truth.dat', 'r') as f:
    truth = f.read().strip()

n = len(truth)
print(f'Truth length: {n}')

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
half = n // 2

print('\n=== Factor analysis ===')
temp = n
factors = []
for p in range(2, 10000):
    while temp % p == 0:
        factors.append(p)
        temp //= p
if temp > 1:
    factors.append(temp)
print(f'n = {" x ".join(str(f) for f in factors)}')

print('\n=== Autocorrelation check ===')
for lag in [4, 8, 16, 21, 32, 168, 336]:
    corr = np.mean(arr[:100000] == arr[lag:100000+lag])
    print(f'Lag {lag}: {corr:.6f}')

print('\n=== Brute force ISCC search in XOR space ===')
raw_bytes = bits_to_bytes(arr)
inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes(inv_bits)

iscc = b'ISCC{'
iscc_arr = np.frombuffer(iscc, dtype=np.uint8)

for data_name, data_bytes in [('raw', raw_bytes), ('inv', inv_bytes)]:
    data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
        key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
        key_len = len(key_arr)
        for offset in range(key_len):
            expected = iscc_arr ^ key_arr[offset:offset+5]
            search_data = data_arr[offset:]
            for pos in range(0, min(len(search_data) - 5, 1000000)):
                if np.array_equal(search_data[pos:pos+5], expected):
                    full_data = search_data[pos:]
                    decrypted = bytes(full_data[i] ^ key_bytes[(offset + i) % key_len] for i in range(min(200, len(full_data))))
                    print(f'Match [{data_name}+{key_name}+off{offset}] pos={pos}: {repr(decrypted[:80])}')
                    if b'ISCC{' in decrypted:
                        end = decrypted.find(b'}', 5)
                        if end >= 0:
                            print(f'*** FLAG: {decrypted[:end+1].decode()} ***')

print('\n=== Try AES-CTR ===')
try:
    from Crypto.Cipher import AES
    from Crypto.Util import Counter
    print('PyCryptodome available')
    for data_name, data_bytes in [('raw', raw_bytes), ('inv', inv_bytes)]:
        for key_name, key_bytes in [('dec16', key[:16].ljust(16,b'\x00')), ('b6416', key_b64[:16])]:
            for nonce in [b'\x00'*8, key[:8]]:
                try:
                    ctr = Counter.new(64, prefix=nonce, initial_value=0)
                    cipher = AES.new(key_bytes, AES.MODE_CTR, counter=ctr)
                    result = cipher.decrypt(data_bytes[:100000])
                    if search_flag(result, f'AES_{data_name}_{key_name}'):
                        sys.exit(0)
                    printable = sum(1 for b in result[:100] if 32 <= b < 127)
                    print(f'AES-CTR {data_name}+{key_name}: printable={printable/100:.2f}')
                except Exception as e:
                    print(f'AES error: {e}')
except ImportError:
    print('PyCryptodome not available')
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        print('cryptography available')
        for data_name, data_bytes in [('raw', raw_bytes), ('inv', inv_bytes)]:
            for key_name, key_bytes in [('dec16', key[:16].ljust(16,b'\x00')), ('b6416', key_b64[:16])]:
                for nonce in [b'\x00'*16, key[:16].ljust(16,b'\x00')[:16]]:
                    try:
                        cipher = Cipher(algorithms.AES(key_bytes), modes.CTR(nonce))
                        dec = cipher.decryptor()
                        result = dec.update(data_bytes[:100000]) + dec.finalize()
                        if search_flag(result, f'AES_{data_name}_{key_name}'):
                            sys.exit(0)
                    except Exception as e:
                        print(f'AES error: {e}')
    except ImportError:
        print('No AES library available')

print('\n=== Try columnar transposition with key ordering ===')
key_str = 'Yth9Ur062azZA09y4rs5L'
col_order_4 = sorted(range(4), key=lambda i: key_str[i])
inv_order_4 = [0]*4
for i, p in enumerate(col_order_4):
    inv_order_4[p] = i
print(f'Key4 order: {col_order_4}, inverse: {inv_order_4}')

mat4 = arr[:n_rows*4].reshape(n_rows, 4)
reordered = mat4[:, inv_order_4]
reordered_bytes = bits_to_bytes(reordered.flatten())
result = xor_key(reordered_bytes, key)
if search_flag(result, 'KeyColReorder+XOR'):
    sys.exit(0)
print(f'KeyColReorder first: {result[:50].hex()}')

print('\n=== Try: Möbius XOR at matrix level with different flip combinations ===')
half_r = n_rows // 2
top = mat4[:half_r]
bottom = mat4[half_r:2*half_r]

combos = [
    ('top XOR flip_ud(flip_lr(not bottom))', top ^ np.flip(np.flip(1-bottom, axis=1), axis=0)),
    ('top XOR flip_ud(not bottom)', top ^ np.flip(1-bottom, axis=0)),
    ('top XOR flip_lr(not bottom)', top ^ np.flip(1-bottom, axis=1)),
    ('top XOR not bottom', top ^ (1-bottom)),
    ('top XOR flip_ud(bottom)', top ^ np.flip(bottom, axis=0)),
    ('top XOR flip_lr(bottom)', top ^ np.flip(bottom, axis=1)),
    ('just top', top),
]

for name, data in combos:
    data_bytes = bits_to_bytes(data.flatten())
    for key_name, key_bytes in [('dec', key), ('b64', key_b64), ('none', b'')]:
        if key_bytes:
            result = xor_key(data_bytes, key_bytes)
        else:
            result = data_bytes
        if search_flag(result, f'{name}+{key_name}'):
            sys.exit(0)

print('\n=== Try: nibble-level operations ===')
n_nibbles = n // 4
nibble_data = arr[:n_nibbles*4].reshape(-1, 4)
nibble_vals = (nibble_data * np.array([8,4,2,1])).sum(axis=1).astype(np.uint8)

nibble_half = len(nibble_vals) // 2
nib_first = nibble_vals[:nibble_half]
nib_second = nibble_vals[nibble_half:2*nibble_half]

for name, data in [
    ('nib_first', nib_first),
    ('nib XOR rev_flip_second', nib_first ^ np.flip(15 - nib_second)),
    ('nib_first_bytes', nib_first),
]:
    data_bytes = data.tobytes()
    for key_name, key_bytes in [('dec', key), ('b64', key_b64), ('none', b'')]:
        if key_bytes:
            result = xor_key(data_bytes, key_bytes)
        else:
            result = data_bytes
        if search_flag(result, f'{name}+{key_name}'):
            sys.exit(0)

if len(nibble_vals) % 2 == 0:
    high = nibble_vals[0::2]
    low = nibble_vals[1::2]
    paired = ((high << 4) | low).tobytes()
    for key_name, key_bytes in [('dec', key), ('b64', key_b64), ('none', b'')]:
        if key_bytes:
            result = xor_key(paired, key_bytes)
        else:
            result = paired
        if search_flag(result, f'nibble_paired+{key_name}'):
            sys.exit(0)

print('\nDone - no flag found')
