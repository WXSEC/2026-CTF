import base64
import numpy as np
import re

with open('secret.dat', 'r') as f:
    secret = f.read().strip()

secret_bin = ''
for c in secret:
    secret_bin += bin(int(c, 16))[2:].zfill(4)

rows = len(secret_bin) // 4
matrix = []
for i in range(rows):
    matrix.append(secret_bin[i*4:(i+1)*4])

col_data = ''
for col in range(4):
    for row in range(rows):
        col_data += matrix[row][col]

result_hex = ''
for i in range(0, len(col_data), 4):
    nibble = col_data[i:i+4]
    result_hex += hex(int(nibble, 2))[2:]

secret_decoded = bytes.fromhex(result_hex)
print(f'Decoded secret: {secret_decoded}')

try:
    key = base64.b64decode(secret_decoded)
    print(f'Key: {key}')
except:
    key = secret_decoded
    print(f'Key (raw): {key}')

print(f'Key length: {len(key)}')

with open('truth.dat', 'r') as f:
    truth = f.read().strip()

n = len(truth)
print(f'Truth length: {n}')

arr = np.frombuffer(truth.encode('ascii'), dtype=np.uint8) - ord('0')
print(f'Array shape: {arr.shape}, min: {arr.min()}, max: {arr.max()}')

def bits_to_bytes_np(bits):
    bits = bits[:len(bits) - len(bits) % 8]
    bits = bits.reshape(-1, 8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
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
    idx = text.find('ISCC')
    if idx >= 0:
        print(f'[{label}] Found ISCC at position {idx}: {repr(text[idx:idx+50])}')
    hex_text = data.hex()
    idx = hex_text.find('49534343')
    if idx >= 0:
        print(f'[{label}] Found ISCC hex at position {idx}: {data[idx//2:idx//2+30]}')
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

print('\n=== Approach 1: Direct bytes + XOR ===')
raw_bytes = bits_to_bytes_np(arr)
print(f'Raw bytes length: {len(raw_bytes)}')
result = xor_key(raw_bytes, key)
search_flag(result, 'Direct+XOR')

print('\n=== Approach 2: Direct bytes + RC4 ===')
result = rc4_decrypt(raw_bytes, key)
search_flag(result, 'Direct+RC4')

print('\n=== Approach 3: 4-col bit transposition (row->col) + XOR ===')
n_rows = n // 4
mat = arr[:n_rows*4].reshape(n_rows, 4)
trans = mat.T.flatten()
trans_bytes = bits_to_bytes_np(trans)
result = xor_key(trans_bytes, key)
search_flag(result, 'BitTrans+XOR')

print('\n=== Approach 4: 4-col bit transposition + RC4 ===')
result = rc4_decrypt(trans_bytes, key)
search_flag(result, 'BitTrans+RC4')

print('\n=== Approach 5: Inverse 4-col bit transposition (col->row) + XOR ===')
inv_mat = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes_np(inv_mat)
result = xor_key(inv_bytes, key)
search_flag(result, 'InvBitTrans+XOR')

print('\n=== Approach 6: Inverse 4-col bit transposition + RC4 ===')
result = rc4_decrypt(inv_bytes, key)
search_flag(result, 'InvBitTrans+RC4')

print('\n=== Approach 7: Byte-level 4-col transposition (row->col) + XOR ===')
n_bytes_total = len(raw_bytes)
n_rows_b = n_bytes_total // 4
byte_mat = np.frombuffer(raw_bytes, dtype=np.uint8)[:n_rows_b*4].reshape(n_rows_b, 4)
trans_bytes2 = byte_mat.T.flatten().tobytes()
result = xor_key(trans_bytes2, key)
search_flag(result, 'ByteTrans+XOR')

print('\n=== Approach 8: Byte-level inverse 4-col transposition + XOR ===')
raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)[:n_rows_b*4]
inv_byte_mat = raw_arr.reshape(4, n_rows_b).T.flatten().tobytes()
result = xor_key(inv_byte_mat, key)
search_flag(result, 'InvByteTrans+XOR')

print('\n=== Approach 9: Transpose + XOR with b64 key string ===')
b64_key = secret_decoded
result = xor_key(trans_bytes, b64_key)
search_flag(result, 'BitTrans+XOR_b64key')

print('\n=== Approach 10: Inverse transpose + XOR with b64 key string ===')
result = xor_key(inv_bytes, b64_key)
search_flag(result, 'InvBitTrans+XOR_b64key')

print('\n=== Approach 11: Nibble-level (4-bit) transposition ===')
n_nibbles = n // 4
nibble_rows = n_nibbles // 4
nibble_mat = arr[:nibble_rows*4*4].reshape(nibble_rows, 4, 4)
for col in range(4):
    nibble_trans = nibble_mat[:, col, :].flatten()
    nb = bits_to_bytes_np(nibble_trans)
    result = xor_key(nb, key)
    if search_flag(result, f'NibbleTrans_col{col}+XOR'):
        break

print('\n=== Approach 12: Bit-level XOR with key bits then transposition ===')
key_bits = ''
for b in key:
    key_bits += bin(b)[2:].zfill(8)
key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
key_tiled = np.tile(key_bit_arr, len(arr) // len(key_bit_arr) + 1)[:len(arr)]
xored = (arr ^ key_tiled)
xored_bytes = bits_to_bytes_np(xored)
search_flag(xored_bytes, 'BitXOR_then_bytes')

xored_trans = xored[:n_rows*4].reshape(n_rows, 4).T.flatten()
xored_trans_bytes = bits_to_bytes_np(xored_trans)
search_flag(xored_trans_bytes, 'BitXOR_then_trans')

print('\n=== Approach 13: Möbius strip - flip second half bits ===')
half = n // 2
first_half = arr[:half]
second_half = arr[half:2*half]
flipped = 1 - second_half
combined = np.concatenate([first_half, flipped])
combined_bytes = bits_to_bytes_np(combined)
result = xor_key(combined_bytes, key)
search_flag(result, 'MobiusFlip+XOR')

combined_trans = combined[:n_rows*4].reshape(n_rows, 4).T.flatten()
combined_trans_bytes = bits_to_bytes_np(combined_trans)
result = xor_key(combined_trans_bytes, key)
search_flag(result, 'MobiusFlip+Trans+XOR')

print('\n=== Approach 14: Möbius strip - flip and reverse second half ===')
reversed_second = np.flip(second_half)
combined2 = np.concatenate([first_half, reversed_second])
combined2_bytes = bits_to_bytes_np(combined2)
result = xor_key(combined2_bytes, key)
search_flag(result, 'MobiusFlipRev+XOR')

print('\n=== Approach 15: LSB-first byte conversion ===')
def bits_to_bytes_lsb(bits):
    bits = bits[:len(bits) - len(bits) % 8]
    bits = bits.reshape(-1, 8)
    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()

lsb_bytes = bits_to_bytes_lsb(arr)
result = xor_key(lsb_bytes, key)
search_flag(result, 'LSB+XOR')

lsb_trans = arr[:n_rows*4].reshape(n_rows, 4).T.flatten()
lsb_trans_bytes = bits_to_bytes_lsb(lsb_trans)
result = xor_key(lsb_trans_bytes, key)
search_flag(result, 'LSB_Trans+XOR')

print('\nDone with approaches 1-15.')
