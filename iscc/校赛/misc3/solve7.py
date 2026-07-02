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

half = n // 2
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
    for pattern in ['ISCC{', 'ISCC', 'flag{', 'Flag{']:
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
            print(f'[{label}] hex ISCC at byte {idx//2}: {data[idx//2:idx//2+30]}')
            return True
    return False

def check(data, label):
    text = data[:200].decode('utf-8', errors='replace')
    printable = sum(1 for b in data[:1000] if 32 <= b <= 126)
    ratio = printable / min(len(data), 1000)
    print(f'[{label}] ratio={ratio:.2f} first80={repr(text[:80])}')
    return ratio

print('=== Key insight: "起点亦或终点" = "起点XOR终点" (XOR start with end) ===')
print()

first_half = arr[:half]
second_half = arr[half:2*half]

print('=== Approach 1: XOR first half with second half ===')
xored_halves = first_half ^ second_half
xh_bytes = bits_to_bytes(xored_halves)
search_flag(xh_bytes, 'XOR_halves')
check(xh_bytes, 'XOR_halves')

print('\n=== Approach 2: XOR first half with reversed second half ===')
xored_rev = first_half ^ np.flip(second_half)
xr_bytes = bits_to_bytes(xored_rev)
search_flag(xr_bytes, 'XOR_halves_rev')
check(xr_bytes, 'XOR_halves_rev')

print('\n=== Approach 3: XOR first half with flipped second half ===')
xored_flip = first_half ^ (1 - second_half)
xf_bytes = bits_to_bytes(xored_flip)
search_flag(xf_bytes, 'XOR_halves_flip')
check(xf_bytes, 'XOR_halves_flip')

print('\n=== Approach 4: XOR first half with reversed+flipped second half ===')
xored_rf = first_half ^ np.flip(1 - second_half)
xrf_bytes = bits_to_bytes(xored_rf)
search_flag(xrf_bytes, 'XOR_halves_revflip')
check(xrf_bytes, 'XOR_halves_revflip')

print('\n=== Approach 5: Reverse entire data, then decode ===')
rev_arr = np.flip(arr)
rev_inv = rev_arr[:n_rows*4].reshape(4, n_rows).T.flatten()
rev_inv_bytes = bits_to_bytes(rev_inv)
result = xor_bytes(rev_inv_bytes, key)
search_flag(result, 'Reverse_InvTrans+XOR')
check(result, 'Reverse_InvTrans+XOR')

rev_direct = bits_to_bytes(rev_arr)
result = xor_bytes(rev_direct, key)
search_flag(result, 'Reverse_Direct+XOR')
check(result, 'Reverse_Direct+XOR')

print('\n=== Approach 6: XOR halves then inverse transposition ===')
for xh, name in [(xored_halves, 'xor'), (xored_rev, 'xor_rev'), 
                  (xored_flip, 'xor_flip'), (xored_rf, 'xor_revflip')]:
    xh_rows = len(xh) // 4
    xh_inv = xh[:xh_rows*4].reshape(4, xh_rows).T.flatten()
    xh_inv_bytes = bits_to_bytes(xh_inv)
    result = xor_bytes(xh_inv_bytes, key)
    if search_flag(result, f'{name}_InvTrans+XOR'):
        sys.exit(0)
    check(result, f'{name}_InvTrans+XOR')

print('\n=== Approach 7: Inverse transposition first, then XOR halves ===')
inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_half = len(inv_bits) // 2
inv_first = inv_bits[:inv_half]
inv_second = inv_bits[inv_half:2*inv_half]

for xh, name in [(inv_first ^ inv_second, 'inv_xor'), 
                  (inv_first ^ np.flip(inv_second), 'inv_xor_rev'),
                  (inv_first ^ (1 - inv_second), 'inv_xor_flip'),
                  (inv_first ^ np.flip(1 - inv_second), 'inv_xor_revflip')]:
    xh_bytes = bits_to_bytes(xh)
    result = xor_bytes(xh_bytes, key)
    if search_flag(result, f'{name}+XOR'):
        sys.exit(0)
    check(result, f'{name}+XOR')

print('\n=== Approach 8: XOR halves at byte level ===')
raw_bytes = bits_to_bytes(arr)
n_b = len(raw_bytes)
b_half = n_b // 2
first_b = np.frombuffer(raw_bytes[:b_half], dtype=np.uint8)
second_b = np.frombuffer(raw_bytes[b_half:2*b_half], dtype=np.uint8)

xored_b = first_b ^ second_b
search_flag(xored_b.tobytes(), 'ByteXOR_halves')
check(xored_b.tobytes(), 'ByteXOR_halves')

xored_b_rev = first_b ^ np.flip(second_b)
search_flag(xored_b_rev.tobytes(), 'ByteXOR_halves_rev')
check(xored_b_rev.tobytes(), 'ByteXOR_halves_rev')

print('\n=== Approach 9: Nibble-level XOR halves ===')
n_nibbles = n // 4
nibble_first = arr[:n_nibbles*2].reshape(-1, 2)
nibble_second = arr[n_nibbles*2:n_nibbles*4].reshape(-1, 2)

for op_name, op_fn in [('xor', lambda a,b: a^b), 
                         ('xor_rev', lambda a,b: a ^ np.flip(b)),
                         ('xor_flip', lambda a,b: a ^ (1-b)),
                         ('xor_revflip', lambda a,b: a ^ np.flip(1-b))]:
    result_nibbles = op_fn(nibble_first, nibble_second)
    result_bits = result_nibbles.flatten()
    result_bytes = bits_to_bytes(result_bits)
    result_final = xor_bytes(result_bytes, key)
    if search_flag(result_final, f'Nibble_{op_name}+XOR'):
        sys.exit(0)
    check(result_final, f'Nibble_{op_name}+XOR')

print('\n=== Approach 10: Full data XOR with key, then check halves ===')
xored_full = xor_bits(arr, key)
xf_first = xored_full[:half]
xf_second = xored_full[half:2*half]

for op_name, op_fn in [('xor', lambda a,b: a^b),
                         ('xor_rev', lambda a,b: a ^ np.flip(b)),
                         ('xor_flip', lambda a,b: a ^ (1-b)),
                         ('xor_revflip', lambda a,b: a ^ np.flip(1-b))]:
    result_bits = op_fn(xf_first, xf_second)
    result_bytes = bits_to_bytes(result_bits)
    if search_flag(result_bytes, f'KeyXOR_{op_name}'):
        sys.exit(0)
    check(result_bytes, f'KeyXOR_{op_name}')

print('\nDone.')
