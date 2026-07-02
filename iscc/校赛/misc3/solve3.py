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
print(f'n/8 = {n/8}')
print(f'n/4 = {n/4}')

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

def bits_to_bytes_lsb(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    rem = len(bits) % 8
    if rem:
        bits = bits[:len(bits) - rem]
    bits = bits.reshape(-1, 8)
    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()

def search_flag(data, label):
    text = data.decode('utf-8', errors='replace')
    idx = text.find('ISCC{')
    if idx >= 0:
        end = text.find('}', idx)
        if end >= 0:
            flag = text[idx:end+1]
            print(f'*** FLAG FOUND [{label}]: {flag} ***')
            return True
        print(f'[{label}] ISCC{{ at {idx}: {repr(text[idx:idx+80])}')
    idx = text.find('ISCC')
    if idx >= 0:
        print(f'[{label}] ISCC at {idx}: {repr(text[idx:idx+50])}')
    hex_text = data.hex()
    idx = hex_text.find('49534343')
    if idx >= 0:
        print(f'[{label}] ISCC hex at {idx}')
    return False

def show_first(data, label, n=50):
    text = data[:n].decode('utf-8', errors='replace')
    print(f'[{label}] First {n} chars: {repr(text)}')
    print(f'[{label}] First {n} bytes hex: {data[:n].hex()}')

def xor_key(data, key_bytes):
    key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    key_tiled = np.tile(key_arr, len(data_arr) // len(key_arr) + 1)[:len(data_arr)]
    return (data_arr ^ key_tiled).tobytes()

half = n // 2
n_rows = n // 4

print(f'\nhalf={half}, n_rows={n_rows}')
print(f'half/8={half/8}, half/4={half/4}')

print('\n=== Check Mobius relationship between halves ===')
first_half = arr[:half]
second_half = arr[half:2*half]
print(f'first == flipped_second: {np.mean(first_half == (1-second_half)):.6f}')
print(f'first == reversed_flipped_second: {np.mean(first_half == np.flip(1-second_half)):.6f}')
print(f'first == reversed_second: {np.mean(first_half == np.flip(second_half)):.6f}')

print('\n=== Check at 4-col matrix level ===')
mat4 = arr[:n_rows*4].reshape(n_rows, 4)
top = mat4[:n_rows//2]
bottom = mat4[n_rows//2:]
print(f'top == flip_lr(bottom): {np.mean(top == np.flip(bottom, axis=1)):.6f}')
print(f'top == flip_ud(bottom): {np.mean(top == np.flip(bottom, axis=0)):.6f}')
print(f'top == flip_both(bottom): {np.mean(top == np.flip(np.flip(bottom, axis=1), axis=0)):.6f}')

print('\n=== Check at column level ===')
for c in range(4):
    col = mat4[:, c]
    col_first = col[:n_rows//2]
    col_second = col[n_rows//2:]
    print(f'Col{c}: first==flip(second): {np.mean(col_first == np.flip(col_second)):.6f}, first==second: {np.mean(col_first == col_second):.6f}')

print('\n=== Approach 1: Inverse transposition (reshape(4,n_rows).T) + XOR ===')
inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes(inv_bits)
show_first(inv_bytes, 'InvTrans_raw')
result = xor_key(inv_bytes, key)
show_first(result, 'InvTrans+XOR')
search_flag(result, 'InvTrans+XOR')

print('\n=== Approach 2: Forward transposition (reshape(n_rows,4).T) + XOR ===')
fwd_bits = arr[:n_rows*4].reshape(n_rows, 4).T.flatten()
fwd_bytes = bits_to_bytes(fwd_bits)
show_first(fwd_bytes, 'FwdTrans_raw')
result = xor_key(fwd_bytes, key)
show_first(result, 'FwdTrans+XOR')
search_flag(result, 'FwdTrans+XOR')

print('\n=== Approach 3: Direct bytes + XOR ===')
raw_bytes = bits_to_bytes(arr)
show_first(raw_bytes, 'Direct_raw')
result = xor_key(raw_bytes, key)
show_first(result, 'Direct+XOR')
search_flag(result, 'Direct+XOR')

print('\n=== Approach 4: Nibble-level (4-bit) transposition ===')
n_nibbles = n // 4
nibble_rows = n_nibbles // 4
nibble_data = arr[:n_nibbles*4].reshape(n_nibbles, 4)
nibble_trans = nibble_data.T.flatten()
nibble_bytes = bits_to_bytes(nibble_trans)
show_first(nibble_bytes, 'NibbleTrans_raw')
result = xor_key(nibble_bytes, key)
show_first(result, 'NibbleTrans+XOR')
search_flag(result, 'NibbleTrans+XOR')

nibble_inv = arr[:n_nibbles*4].reshape(4, n_nibbles).T.flatten()
nibble_inv_bytes = bits_to_bytes(nibble_inv)
result = xor_key(nibble_inv_bytes, key)
show_first(result, 'NibbleInvTrans+XOR')
search_flag(result, 'NibbleInvTrans+XOR')

print('\n=== Approach 5: Mobius - inverse transposition then take first half ===')
inv_half_bits = inv_bits[:half]
inv_half_bytes = bits_to_bytes(inv_half_bits)
result = xor_key(inv_half_bytes, key)
show_first(result, 'InvTrans_Half+XOR')
search_flag(result, 'InvTrans_Half+XOR')

print('\n=== Approach 6: Mobius - take first half then inverse transposition ===')
fh = arr[:half]
fh_rows = half // 4
fh_inv = fh[:fh_rows*4].reshape(4, fh_rows).T.flatten()
fh_inv_bytes = bits_to_bytes(fh_inv)
result = xor_key(fh_inv_bytes, key)
show_first(result, 'FH_InvTrans+XOR')
search_flag(result, 'FH_InvTrans+XOR')

print('\n=== Approach 7: Mobius untwist then inverse transposition ===')
mat4 = arr[:n_rows*4].reshape(n_rows, 4)
top_rows = n_rows // 2
top = mat4[:top_rows].copy()
bottom = mat4[top_rows:].copy()
bottom_untwisted = np.flip(np.flip(bottom, axis=1), axis=0)
untwisted = np.vstack([top, bottom_untwisted])
untwisted_inv = untwisted.reshape(n_rows, 4)  # already in row form
untwisted_bytes = bits_to_bytes(untwisted_inv.flatten())
result = xor_key(untwisted_bytes, key)
show_first(result, 'Untwist+XOR')
search_flag(result, 'Untwist+XOR')

untwisted_inv2 = untwisted.flatten().reshape(4, n_rows).T.flatten()
untwisted_inv2_bytes = bits_to_bytes(untwisted_inv2)
result = xor_key(untwisted_inv2_bytes, key)
show_first(result, 'Untwist+InvTrans+XOR')
search_flag(result, 'Untwist+InvTrans+XOR')

print('\n=== Approach 8: XOR at bit level then inverse transposition ===')
key_bits = ''.join(bin(b)[2:].zfill(8) for b in key)
key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
key_tiled = np.tile(key_bit_arr, len(arr) // len(key_bit_arr) + 1)[:len(arr)]
xored = arr ^ key_tiled

xored_inv = xored[:n_rows*4].reshape(4, n_rows).T.flatten()
xored_inv_bytes = bits_to_bytes(xored_inv)
show_first(xored_inv_bytes, 'BitXOR+InvTrans')
search_flag(xored_inv_bytes, 'BitXOR+InvTrans')

print('\n=== Approach 9: Byte XOR then inverse byte transposition ===')
xored_bytes = xor_key(raw_bytes, key)
xa = np.frombuffer(xored_bytes, dtype=np.uint8)
n_b = len(xa)
n_rb = n_b // 4
ibt = xa[:n_rb*4].reshape(4, n_rb).T.flatten().tobytes()
show_first(ibt, 'XOR+InvByteTrans')
search_flag(ibt, 'XOR+InvByteTrans')

print('\n=== Approach 10: Try with b64 key string ===')
b64_key = key_b64
result = xor_key(inv_bytes, b64_key)
show_first(result, 'InvTrans+XORb64')
search_flag(result, 'InvTrans+XORb64')

print('\n=== Approach 11: LSB-first byte conversion ===')
lsb_bytes = bits_to_bytes_lsb(arr)
result = xor_key(lsb_bytes, key)
show_first(result, 'LSB+XOR')
search_flag(result, 'LSB+XOR')

lsb_inv = bits_to_bytes_lsb(inv_bits)
result = xor_key(lsb_inv, key)
show_first(result, 'LSB_InvTrans+XOR')
search_flag(result, 'LSB_InvTrans+XOR')

print('\n=== Approach 12: Mobius strip - read as continuous loop ===')
mat4 = arr[:n_rows*4].reshape(n_rows, 4)
loop_bits = []
for row in range(n_rows):
    if row < n_rows // 2:
        loop_bits.append(mat4[row])
    else:
        loop_bits.append(np.flip(mat4[row]))

loop_arr = np.array(loop_bits).flatten()
loop_bytes = bits_to_bytes(loop_arr)
result = xor_key(loop_bytes, key)
show_first(result, 'Loop+XOR')
search_flag(result, 'Loop+XOR')

print('\n=== Approach 13: Mobius strip - snake reading ===')
snake_bits = []
for row in range(n_rows):
    if row % 2 == 0:
        snake_bits.append(mat4[row])
    else:
        snake_bits.append(np.flip(mat4[row]))

snake_arr = np.array(snake_bits).flatten()
snake_bytes = bits_to_bytes(snake_arr)
result = xor_key(snake_bytes, key)
show_first(result, 'Snake+XOR')
search_flag(result, 'Snake+XOR')

print('\nDone with approaches 1-13.')
