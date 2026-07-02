import base64
import numpy as np
import sys
import io
import struct

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

def bits_to_bytes_lsb(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    rem = len(bits) % 8
    if rem:
        bits = bits[:len(bits) - rem]
    bits = bits.reshape(-1, 8)
    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()

def xor_key(data, key_bytes):
    key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    key_tiled = np.tile(key_arr, len(data_arr) // len(key_arr) + 1)[:len(data_arr)]
    return (data_arr ^ key_tiled).tobytes()

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

def rc4_decrypt(data, key_bytes):
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

half = n // 2
n_rows = n // 4

print(f'\nhalf={half}, n_rows={n_rows}')

print('\n' + '='*60)
print('MOBIUS STRIP MODEL: Start XOR End')
print('Return = spatial reversal (reverse) + directional reversal (bit flip)')
print('='*60)

first_half = arr[:half]
second_half = arr[half:2*half]

print('\n=== M1: XOR first_half with reverse(flip(second_half)) ===')
mobius_second = np.flip(1 - second_half)
result_bits = first_half ^ mobius_second
result_bytes = bits_to_bytes(result_bits)
show_first(result_bytes, 'M1_raw')
search_flag(result_bytes, 'M1_raw')
result = xor_key(result_bytes, key)
show_first(result, 'M1+XOR')
search_flag(result, 'M1+XOR')

print('\n=== M2: XOR first_half with flip(reverse(second_half)) ===')
mobius_second2 = 1 - np.flip(second_half)
result_bits2 = first_half ^ mobius_second2
result_bytes2 = bits_to_bytes(result_bits2)
show_first(result_bytes2, 'M2_raw')
search_flag(result_bytes2, 'M2_raw')
result = xor_key(result_bytes2, key)
show_first(result, 'M2+XOR')
search_flag(result, 'M2+XOR')

print('\n=== M3: Just take first_half (second = reverse(flip(first))) ===')
fh_bytes = bits_to_bytes(first_half)
show_first(fh_bytes, 'M3_raw')
search_flag(fh_bytes, 'M3_raw')
result = xor_key(fh_bytes, key)
show_first(result, 'M3+XOR')
search_flag(result, 'M3+XOR')

print('\n=== M4: Inverse transpose first, then Mobius untwist ===')
inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_first = inv_bits[:half]
inv_second = inv_bits[half:2*half]

inv_mobius = np.flip(1 - inv_second)
inv_result = inv_first ^ inv_mobius
inv_result_bytes = bits_to_bytes(inv_result)
show_first(inv_result_bytes, 'M4_InvTrans_Mobius')
search_flag(inv_result_bytes, 'M4_InvTrans_Mobius')
result = xor_key(inv_result_bytes, key)
show_first(result, 'M4_InvTrans_Mobius+XOR')
search_flag(result, 'M4_InvTrans_Mobius+XOR')

print('\n=== M5: Mobius untwist first, then inverse transpose ===')
mobius_data = first_half ^ np.flip(1 - second_half)
mobius_inv = mobius_data[:n_rows*2].reshape(2, n_rows).T.flatten()
mobius_inv_bytes = bits_to_bytes(mobius_inv)
show_first(mobius_inv_bytes, 'M5_Mobius_InvTrans')
search_flag(mobius_inv_bytes, 'M5_Mobius_InvTrans')
result = xor_key(mobius_inv_bytes, key)
show_first(result, 'M5_Mobius_InvTrans+XOR')
search_flag(result, 'M5_Mobius_InvTrans+XOR')

print('\n=== M6: Take first_half after inverse transpose ===')
inv_fh = inv_bits[:len(inv_bits)//2]
inv_fh_bytes = bits_to_bytes(inv_fh)
result = xor_key(inv_fh_bytes, key)
show_first(result, 'M6_InvTrans_FirstHalf+XOR')
search_flag(result, 'M6_InvTrans_FirstHalf+XOR')

print('\n=== M7: RC4 decrypt direct bytes ===')
raw_bytes = bits_to_bytes(arr)
rc4_result = rc4_decrypt(raw_bytes, key)
show_first(rc4_result, 'M7_RC4_direct')
search_flag(rc4_result, 'M7_RC4_direct')

print('\n=== M8: RC4 decrypt inverse transposed bytes ===')
inv_bytes = bits_to_bytes(inv_bits)
rc4_result = rc4_decrypt(inv_bytes, key)
show_first(rc4_result, 'M8_RC4_InvTrans')
search_flag(rc4_result, 'M8_RC4_InvTrans')

print('\n=== M9: RC4 with b64 key string ===')
rc4_result = rc4_decrypt(raw_bytes, key_b64)
show_first(rc4_result, 'M9_RC4_b64key')
search_flag(rc4_result, 'M9_RC4_b64key')

rc4_result = rc4_decrypt(inv_bytes, key_b64)
show_first(rc4_result, 'M9_RC4_InvTrans_b64key')
search_flag(rc4_result, 'M9_RC4_InvTrans_b64key')

print('\n=== M10: Mobius untwist at nibble level ===')
n_nibbles = n // 4
nibble_mat = arr[:n_nibbles*4].reshape(n_nibbles, 4)
nib_first = nibble_mat[:n_nibbles//2]
nib_second = nibble_mat[n_nibbles//2:]
nib_mobius = np.flip(np.flip(1 - nib_second, axis=1), axis=0)
nib_result = nib_first ^ nib_mobius
nib_result_bits = nib_result.flatten()
nib_result_bytes = bits_to_bytes(nib_result_bits)
show_first(nib_result_bytes, 'M10_NibbleMobius')
search_flag(nib_result_bytes, 'M10_NibbleMobius')
result = xor_key(nib_result_bytes, key)
show_first(result, 'M10_NibbleMobius+XOR')
search_flag(result, 'M10_NibbleMobius+XOR')

print('\n=== M11: Full Mobius model - 4-col matrix, top/bottom halves ===')
mat4 = arr[:n_rows*4].reshape(n_rows, 4)
top_rows = n_rows // 2
top = mat4[:top_rows]
bottom = mat4[top_rows:]

bottom_untwisted = np.flip(np.flip(1 - bottom, axis=1), axis=0)
untwisted = top ^ bottom_untwisted
untwisted_bytes = bits_to_bytes(untwisted.flatten())
show_first(untwisted_bytes, 'M4_MatrixMobius')
search_flag(untwisted_bytes, 'M4_MatrixMobius')
result = xor_key(untwisted_bytes, key)
show_first(result, 'M4_MatrixMobius+XOR')
search_flag(result, 'M4_MatrixMobius+XOR')

print('\n=== M12: Matrix Mobius then inverse transpose ===')
untwisted_inv = untwisted.flatten().reshape(top_rows, 4).T.flatten()
untwisted_inv_bytes = bits_to_bytes(untwisted_inv)
show_first(untwisted_inv_bytes, 'M12_MatMobius_InvTrans')
search_flag(untwisted_inv_bytes, 'M12_MatMobius_InvTrans')
result = xor_key(untwisted_inv_bytes, key)
show_first(result, 'M12_MatMobius_InvTrans+XOR')
search_flag(result, 'M12_MatMobius_InvTrans+XOR')

print('\n=== M13: Matrix Mobius with only row flip (no col flip) ===')
bottom_untwisted2 = np.flip(1 - bottom, axis=0)
untwisted2 = top ^ bottom_untwisted2
untwisted2_bytes = bits_to_bytes(untwisted2.flatten())
show_first(untwisted2_bytes, 'M13_RowFlipOnly')
search_flag(untwisted2_bytes, 'M13_RowFlipOnly')
result = xor_key(untwisted2_bytes, key)
show_first(result, 'M13_RowFlipOnly+XOR')
search_flag(result, 'M13_RowFlipOnly+XOR')

print('\n=== M14: Matrix Mobius with only col flip (no row flip) ===')
bottom_untwisted3 = np.flip(1 - bottom, axis=1)
untwisted3 = top ^ bottom_untwisted3
untwisted3_bytes = bits_to_bytes(untwisted3.flatten())
show_first(untwisted3_bytes, 'M14_ColFlipOnly')
search_flag(untwisted3_bytes, 'M14_ColFlipOnly')
result = xor_key(untwisted3_bytes, key)
show_first(result, 'M14_ColFlipOnly+XOR')
search_flag(result, 'M14_ColFlipOnly+XOR')

print('\n=== M15: XOR first_half with second_half (no flip/reverse) ===')
xor_halves = first_half ^ second_half
xor_halves_bytes = bits_to_bytes(xor_halves)
show_first(xor_halves_bytes, 'M15_XorHalves')
search_flag(xor_halves_bytes, 'M15_XorHalves')
result = xor_key(xor_halves_bytes, key)
show_first(result, 'M15_XorHalves+XOR')
search_flag(result, 'M15_XorHalves+XOR')

print('\n=== M16: XOR first_half with reversed second_half ===')
xor_rev = first_half ^ np.flip(second_half)
xor_rev_bytes = bits_to_bytes(xor_rev)
show_first(xor_rev_bytes, 'M16_XorRevHalves')
search_flag(xor_rev_bytes, 'M16_XorRevHalves')
result = xor_key(xor_rev_bytes, key)
show_first(result, 'M16_XorRevHalves+XOR')
search_flag(result, 'M16_XorRevHalves+XOR')

print('\n=== M17: XOR first_half with flipped second_half ===')
xor_flip = first_half ^ (1 - second_half)
xor_flip_bytes = bits_to_bytes(xor_flip)
show_first(xor_flip_bytes, 'M17_XorFlipHalves')
search_flag(xor_flip_bytes, 'M17_XorFlipHalves')
result = xor_key(xor_flip_bytes, key)
show_first(result, 'M17_XorFlipHalves+XOR')
search_flag(result, 'M17_XorFlipHalves+XOR')

print('\n=== M18: Known-plaintext attack - search for ISCC{ pattern ===')
iscc_bits = ''.join(bin(b)[2:].zfill(8) for b in b'ISCC{')
iscc_arr = np.frombuffer(iscc_bits.encode('ascii'), dtype=np.uint8) - ord('0')
print(f'ISCC{{ bits: {iscc_bits} ({len(iscc_arr)} bits)')

key_bits = ''.join(bin(b)[2:].zfill(8) for b in key)
key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
print(f'Key bits length: {len(key_bit_arr)}')

for transform_name, data_bits in [('raw', arr), ('inv_trans', inv_bits)]:
    data_bytes = bits_to_bytes(data_bits)
    data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    for key_name, key_bytes in [('decoded_key', key), ('b64_key', key_b64)]:
        key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
        key_len = len(key_arr)
        for offset in range(key_len):
            xored = np.bitwise_xor(data_arr[offset:offset+5], key_arr[offset % key_len:offset % key_len + 5])
            if bytes(xored) == b'ISCC{':
                full_xor = xor_key(data_bytes[offset:], key_bytes)
                text = full_xor.decode('utf-8', errors='replace')
                end = text.find('}', 5)
                if end >= 0:
                    flag = 'ISCC{' + text[5:end+1]
                    print(f'*** FLAG FOUND [{transform_name}+{key_name}+offset={offset}]: {flag} ***')

print('\n=== M19: Try with different transpose widths ===')
for width in [2, 8, 16, 21, 32]:
    if n % width != 0:
        print(f'Width {width}: n % width = {n % width}, skipping')
        continue
    rows_w = n // width
    mat_w = arr[:rows_w*width].reshape(rows_w, width)
    inv_w = mat_w.T.flatten()
    inv_w_bytes = bits_to_bytes(inv_w)
    result = xor_key(inv_w_bytes, key)
    if search_flag(result, f'M19_InvTrans_w{width}+XOR'):
        sys.exit(0)
    show_first(result, f'M19_InvTrans_w{width}+XOR', 20)

print('\n=== M20: Byte-level operations ===')
raw_bytes = bits_to_bytes(arr)
raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
n_bytes = len(raw_arr)

for width in [2, 4, 8, 21]:
    if n_bytes % width != 0:
        print(f'Byte width {width}: n_bytes % width = {n_bytes % width}, skipping')
        continue
    rows_b = n_bytes // width
    mat_b = raw_arr[:rows_b*width].reshape(rows_b, width)
    inv_b = mat_b.T.flatten().tobytes()
    result = xor_key(inv_b, key)
    if search_flag(result, f'M20_ByteInvTrans_w{width}+XOR'):
        sys.exit(0)
    rc4_r = rc4_decrypt(inv_b, key)
    if search_flag(rc4_r, f'M20_ByteInvTrans_w{width}+RC4'):
        sys.exit(0)

print('\n=== M21: Möbius + Transpose combinations ===')
for mobius_name, mobius_data in [
    ('XorRevFlip', first_half ^ np.flip(1 - second_half)),
    ('XorFlipRev', first_half ^ (1 - np.flip(second_half))),
    ('XorRev', first_half ^ np.flip(second_half)),
    ('XorFlip', first_half ^ (1 - second_half)),
    ('FirstHalf', first_half),
]:
    mh = len(mobius_data)
    for width in [4, 8]:
        if mh % width != 0:
            continue
        mr = mh // width
        m_mat = mobius_data[:mr*width].reshape(mr, width)
        m_inv = m_mat.T.flatten()
        m_bytes = bits_to_bytes(m_inv)
        result = xor_key(m_bytes, key)
        if search_flag(result, f'M21_{mobius_name}_InvTrans_w{width}+XOR'):
            sys.exit(0)
        rc4_r = rc4_decrypt(m_bytes, key)
        if search_flag(rc4_r, f'M21_{mobius_name}_InvTrans_w{width}+RC4'):
            sys.exit(0)

print('\n=== M22: Check if data has repeating pattern with period = key_bit_length ===')
key_bit_len = len(key_bit_arr)
print(f'Key bit length: {key_bit_len}')
periods_to_check = [key_bit_len, key_bit_len * 2, 168, 336]
for period in periods_to_check:
    n_periods = n // period
    if n_periods < 2:
        continue
    chunks = arr[:n_periods*period].reshape(n_periods, period)
    mean_corr = np.mean([np.mean(chunks[0] == chunks[i]) for i in range(1, min(100, n_periods))])
    print(f'Period {period}: mean correlation with first chunk = {mean_corr:.6f}')

print('\nDone!')
