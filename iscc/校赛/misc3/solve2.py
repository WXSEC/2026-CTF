import base64
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('secret.dat', 'r') as f:
    secret = f.read().strip()

secret_bin = ''
for c in secret:
    secret_bin += bin(int(c, 16))[2:].zfill(4)

rows_s = len(secret_bin) // 4
matrix_s = []
for i in range(rows_s):
    matrix_s.append(secret_bin[i*4:(i+1)*4])

col_data = ''
for col in range(4):
    for row in range(rows_s):
        col_data += matrix_s[row][col]

result_hex = ''
for i in range(0, len(col_data), 4):
    nibble = col_data[i:i+4]
    result_hex += hex(int(nibble, 2))[2:]

secret_decoded = bytes.fromhex(result_hex)
secret_text = secret_decoded.decode('utf-8', errors='replace')
print(f'Secret decoded text: {secret_text}')

key_b64 = b'WXRoOVVyMDYyYXpaQTA5eTRyczVM'
key = base64.b64decode(key_b64)
print(f'Key: {key}')
print(f'Key length: {len(key)}')

with open('truth.dat', 'r') as f:
    truth = f.read().strip()

n = len(truth)
print(f'Truth length: {n}')

arr = np.frombuffer(truth.encode('ascii'), dtype=np.uint8) - ord('0')
print(f'Array: min={arr.min()}, max={arr.max()}')

def bits_to_bytes_np(bits):
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
    text = None
    try:
        text = data.decode('utf-8', errors='replace')
    except:
        text = str(data)
    idx = text.find('ISCC{')
    if idx >= 0:
        end = text.find('}', idx)
        if end >= 0:
            flag = text[idx:end+1]
            print(f'*** FLAG FOUND [{label}]: {flag} ***')
            return True
        print(f'[{label}] Found ISCC{{ at {idx}: {repr(text[idx:idx+80])}')
    idx = text.find('ISCC')
    if idx >= 0:
        print(f'[{label}] Found ISCC at {idx}: {repr(text[idx:idx+50])}')
    hex_text = data.hex()
    idx = hex_text.find('49534343')
    if idx >= 0:
        print(f'[{label}] Found ISCC hex at {idx}: {data[idx//2:idx//2+30]}')
    return False

def xor_key(data, key_bytes):
    key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    key_tiled = np.tile(key_arr, len(data_arr) // len(key_arr) + 1)[:len(data_arr)]
    return (data_arr ^ key_tiled).tobytes()

def rc4_decrypt(data, key_bytes):
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
        k = S[(S[i] + S[j]) % 256]
        result.append(byte ^ k)
    return bytes(result)

print('\n=== Verify Mobius strip: check first half vs second half ===')
half = n // 2
first_half = arr[:half]
second_half = arr[half:2*half]

flipped_second = 1 - second_half
reversed_flipped_second = np.flip(flipped_second)
reversed_second = np.flip(second_half)

match1 = np.sum(first_half == flipped_second) / half
match2 = np.sum(first_half == reversed_flipped_second) / half
match3 = np.sum(first_half == reversed_second) / half
print(f'First half == flipped second: {match1:.4f}')
print(f'First half == reversed+flipped second: {match2:.4f}')
print(f'First half == reversed second: {match3:.4f}')

n_rows = n // 4
print(f'\nn_rows = {n_rows}')

print('\n=== Approach A: Mobius column reading (Col0 down, Col3 up, Col1 down, Col2 up) ===')
seg_len = n_rows
col0 = arr[0:seg_len]
col3_rev = arr[seg_len:2*seg_len]
col3 = np.flip(col3_rev)
col1 = arr[2*seg_len:3*seg_len]
col2_rev = arr[3*seg_len:4*seg_len]
col2 = np.flip(col2_rev)

mobius_mat = np.column_stack([col0, col1, col2, col3])
mobius_bits = mobius_mat.flatten()
mobius_bytes = bits_to_bytes_np(mobius_bits)
result = xor_key(mobius_bytes, key)
search_flag(result, 'MobiusCol_A+XOR')
result = rc4_decrypt(mobius_bytes, key)
search_flag(result, 'MobiusCol_A+RC4')

print('\n=== Approach B: Mobius column reading with LSB ===')
mobius_bytes_lsb = bits_to_bytes_lsb(mobius_bits)
result = xor_key(mobius_bytes_lsb, key)
search_flag(result, 'MobiusCol_A+LSB+XOR')

print('\n=== Approach C: Read 4-col matrix with Mobius twist ===')
mat4 = arr[:n_rows*4].reshape(n_rows, 4)
col0_c = mat4[:, 0]
col1_c = mat4[:, 1]
col2_c = mat4[:, 2]
col3_c = mat4[:, 3]

mobius_read = np.concatenate([col0_c, np.flip(col3_c), col1_c, np.flip(col2_c)])
mobius_bytes_c = bits_to_bytes_np(mobius_read)
result = xor_key(mobius_bytes_c, key)
search_flag(result, 'MobiusRead_C+XOR')

print('\n=== Approach D: First half only + transposition ===')
first_half_bits = arr[:half]
fh_rows = half // 4
fh_mat = first_half_bits[:fh_rows*4].reshape(fh_rows, 4)
fh_trans = fh_mat.T.flatten()
fh_bytes = bits_to_bytes_np(fh_trans)
result = xor_key(fh_bytes, key)
search_flag(result, 'FirstHalf_Trans+XOR')

fh_inv = first_half_bits[:fh_rows*4].reshape(4, fh_rows).T.flatten()
fh_inv_bytes = bits_to_bytes_np(fh_inv)
result = xor_key(fh_inv_bytes, key)
search_flag(result, 'FirstHalf_InvTrans+XOR')

print('\n=== Approach E: Full inverse transposition + first half bytes ===')
inv_mat = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes_np(inv_mat)
half_bytes = len(inv_bytes) // 2
result = xor_key(inv_bytes[:half_bytes], key)
search_flag(result, 'InvTrans_FirstHalfBytes+XOR')

print('\n=== Approach F: Snake pattern on 4-col matrix ===')
mat4 = arr[:n_rows*4].reshape(n_rows, 4)
snake_bits = np.array([], dtype=np.uint8)
for row in range(n_rows):
    if row % 2 == 0:
        snake_bits = np.concatenate([snake_bits, mat4[row]])
    else:
        snake_bits = np.concatenate([snake_bits, np.flip(mat4[row])])
snake_bytes = bits_to_bytes_np(snake_bits)
result = xor_key(snake_bytes, key)
search_flag(result, 'Snake+XOR')

print('\n=== Approach G: Mobius strip - unflip bottom half ===')
mat4 = arr[:n_rows*4].reshape(n_rows, 4)
top = mat4[:n_rows//2]
bottom = mat4[n_rows//2:]
bottom_flipped = np.flip(bottom, axis=1)
bottom_reversed = np.flip(bottom, axis=0)
bottom_both = np.flip(np.flip(bottom, axis=1), axis=0)

reconstructed = np.vstack([top, bottom_flipped])
rec_bytes = bits_to_bytes_np(reconstructed.flatten())
result = xor_key(rec_bytes, key)
search_flag(result, 'MobiusUnflip+XOR')

reconstructed2 = np.vstack([top, bottom_reversed])
rec2_bytes = bits_to_bytes_np(reconstructed2.flatten())
result = xor_key(rec2_bytes, key)
search_flag(result, 'MobiusUnreverse+XOR')

reconstructed3 = np.vstack([top, bottom_both])
rec3_bytes = bits_to_bytes_np(reconstructed3.flatten())
result = xor_key(rec3_bytes, key)
search_flag(result, 'MobiusBoth+XOR')

print('\n=== Approach H: XOR first, then byte transposition ===')
raw_bytes = bits_to_bytes_np(arr)
xored = xor_key(raw_bytes, key)
xored_arr = np.frombuffer(xored, dtype=np.uint8)

n_bytes = len(xored)
n_rows_b = n_bytes // 4
byte_mat = xored_arr[:n_rows_b*4].reshape(n_rows_b, 4)
trans_bytes = byte_mat.T.flatten().tobytes()
search_flag(trans_bytes, 'XOR_then_ByteTrans')

inv_byte_mat = xored_arr[:n_rows_b*4].reshape(4, n_rows_b).T.flatten().tobytes()
search_flag(inv_byte_mat, 'XOR_then_InvByteTrans')

print('\n=== Approach I: Bit-level XOR then inverse transposition ===')
key_bits = ''
for b in key:
    key_bits += bin(b)[2:].zfill(8)
key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
key_tiled = np.tile(key_bit_arr, len(arr) // len(key_bit_arr) + 1)[:len(arr)]
xored_bits = (arr ^ key_tiled)

xored_inv = xored_bits[:n_rows*4].reshape(4, n_rows).T.flatten()
xored_inv_bytes = bits_to_bytes_np(xored_inv)
search_flag(xored_inv_bytes, 'BitXOR_then_InvTrans')

xored_inv_lsb = bits_to_bytes_lsb(xored_inv)
search_flag(xored_inv_lsb, 'BitXOR_then_InvTrans_LSB')

print('\n=== Approach J: Key column reordering ===')
key4 = key_b64[:4].decode()
print(f'Key first 4 chars: {key4}')
sorted_pairs = sorted(enumerate(key4), key=lambda x: x[1])
col_order = [p[0] for p in sorted_pairs]
print(f'Column order: {col_order}')

mat4 = arr[:n_rows*4].reshape(n_rows, 4)
reordered = mat4[:, col_order]
re_bytes = bits_to_bytes_np(reordered.flatten())
result = xor_key(re_bytes, key)
search_flag(result, 'KeyColOrder+XOR')

print('\n=== Approach K: Mobius strip as continuous strip ===')
strip_len = n
half_strip = strip_len // 2
first = arr[:half_strip]
second = arr[half_strip:]

second_flipped = 1 - second
second_rev = np.flip(second)
second_rev_flip = np.flip(1 - second)

for name, combined in [('flip', np.concatenate([first, second_flipped])),
                        ('rev', np.concatenate([first, second_rev])),
                        ('revflip', np.concatenate([first, second_rev_flip]))]:
    cb_rows = len(combined) // 4
    cb_mat = combined[:cb_rows*4].reshape(cb_rows, 4)
    cb_trans = cb_mat.T.flatten()
    cb_bytes = bits_to_bytes_np(cb_trans)
    result = xor_key(cb_bytes, key)
    if search_flag(result, f'Strip_{name}+Trans+XOR'):
        break
    cb_inv = combined[:cb_rows*4].reshape(4, cb_rows).T.flatten()
    cb_inv_bytes = bits_to_bytes_np(cb_inv)
    result = xor_key(cb_inv_bytes, key)
    if search_flag(result, f'Strip_{name}+InvTrans+XOR'):
        break

print('\n=== Approach L: First half direct bytes + XOR ===')
first_half_bytes = bits_to_bytes_np(first_half)
result = xor_key(first_half_bytes, key)
search_flag(result, 'FirstHalf_Direct+XOR')

print('\n=== Approach M: RC4 on first half ===')
result = rc4_decrypt(first_half_bytes, key)
search_flag(result, 'FirstHalf_Direct+RC4')

print('\nDone with Mobius approaches.')
