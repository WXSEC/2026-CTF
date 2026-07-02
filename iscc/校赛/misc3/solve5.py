import base64
import numpy as np
import sys
import io
from itertools import permutations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

key_b64 = b'WXRoOVVyMDYyYXpaQTA5eTRyczVM'
key = base64.b64decode(key_b64)

with open('truth.dat', 'r') as f:
    truth = f.read().strip()

n = len(truth)
arr = np.frombuffer(truth.encode('ascii'), dtype=np.uint8) - ord('0')
del truth

n_rows = n // 4
seg = n_rows

CHECK_BITS = 40000

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

def check_flag(data, label):
    text = data.decode('utf-8', errors='replace')
    idx = text.find('ISCC{')
    if idx >= 0:
        end = text.find('}', idx)
        if end >= 0:
            print(f'*** FLAG FOUND [{label}]: {text[idx:end+1]} ***')
            return True
        print(f'[{label}] ISCC{{ at {idx}: {repr(text[idx:idx+80])}')
    idx = text.find('ISCC')
    if idx >= 0:
        print(f'[{label}] ISCC at {idx}: {repr(text[idx:idx+50])}')
    return False

print('=== Phase 1: Column permutations with reversal (check first 5KB) ===')
check_rows = CHECK_BITS // 4
segments = [arr[i*seg:i*seg+check_rows] for i in range(4)]

found = False
for perm in permutations(range(4)):
    if found: break
    for rev_mask in range(16):
        cols = []
        for i in range(4):
            col = segments[perm[i]].copy()
            if rev_mask & (1 << i):
                col = np.flip(col)
            cols.append(col)
        mat = np.column_stack(cols)
        bits = mat.flatten()
        bdata = bits_to_bytes(bits)
        result = xor_bytes(bdata, key)
        if check_flag(result, f'perm={perm}_rev={rev_mask:#06b}'):
            found = True
            break

if not found:
    print('\n=== Phase 2: Bit-XOR first, then column permutations ===')
    xored = xor_bits(arr[:CHECK_BITS*4], key)
    xored_segs = [xored[i*seg:i*seg+check_rows] for i in range(4)]
    
    for perm in permutations(range(4)):
        if found: break
        for rev_mask in range(16):
            cols = []
            for i in range(4):
                col = xored_segs[perm[i]].copy()
                if rev_mask & (1 << i):
                    col = np.flip(col)
            cols.append(col)
            mat = np.column_stack(cols)
            bits = mat.flatten()
            bdata = bits_to_bytes(bits)
            if check_flag(bdata, f'bitxor_perm={perm}_rev={rev_mask:#06b}'):
                found = True
                break

if not found:
    print('\n=== Phase 3: Inverse transposition (full) with different keys ===')
    inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
    inv_bytes = bits_to_bytes(inv_bits)
    
    for k, kn in [(key, 'decoded'), (key_b64, 'b64'), (b'Yth9Ur062azZA09y4rs5L', 'str')]:
        result = xor_bytes(inv_bytes, k)
        if check_flag(result, f'InvTrans+XOR_{kn}'):
            found = True
            break

if not found:
    print('\n=== Phase 4: LSB byte order ===')
    inv_lsb = bits_to_bytes_lsb(inv_bits)
    for k, kn in [(key, 'decoded'), (key_b64, 'b64')]:
        result = xor_bytes(inv_lsb, k)
        if check_flag(result, f'InvTrans_LSB+XOR_{kn}'):
            found = True
            break

if not found:
    print('\n=== Phase 5: Mobius untwist at bit level ===')
    half = n // 2
    first = arr[:half]
    second = arr[half:]
    
    for flip in [False, True]:
        if found: break
        for rev in [False, True]:
            mod = second.copy()
            if flip: mod = 1 - mod
            if rev: mod = np.flip(mod)
            combined = np.concatenate([first, mod])
            c_rows = len(combined) // 4
            inv_c = combined[:c_rows*4].reshape(4, c_rows).T.flatten()
            inv_c_bytes = bits_to_bytes(inv_c)
            result = xor_bytes(inv_c_bytes, key)
            if check_flag(result, f'Mobius_f={flip}_r={rev}+InvTrans+XOR'):
                found = True
                break

if not found:
    print('\n=== Phase 6: Mobius at matrix level ===')
    mat4 = arr[:n_rows*4].reshape(n_rows, 4)
    top_r = n_rows // 2
    top = mat4[:top_r]
    bottom = mat4[top_r:]
    
    for flip_lr in [False, True]:
        if found: break
        for flip_ud in [False, True]:
            bmod = bottom.copy()
            if flip_lr: bmod = np.flip(bmod, axis=1)
            if flip_ud: bmod = np.flip(bmod, axis=0)
            untwisted = np.vstack([top, bmod])
            untwisted_inv = untwisted.flatten().reshape(4, n_rows).T.flatten()
            untwisted_bytes = bits_to_bytes(untwisted_inv)
            result = xor_bytes(untwisted_bytes, key)
            if check_flag(result, f'MatMobius_lr={flip_lr}_ud={flip_ud}+InvTrans+XOR'):
                found = True
                break

if not found:
    print('\n=== Phase 7: Full data search with direct XOR ===')
    raw_bytes = bits_to_bytes(arr)
    result = xor_bytes(raw_bytes, key)
    check_flag(result, 'Direct+XOR_full')

if not found:
    print('\n=== Phase 8: Byte-level transposition ===')
    n_b = len(raw_bytes)
    n_rb = n_b // 4
    
    for direction in ['forward', 'inverse']:
        if direction == 'forward':
            bt = np.frombuffer(raw_bytes, dtype=np.uint8)[:n_rb*4].reshape(n_rb, 4).T.flatten().tobytes()
        else:
            bt = np.frombuffer(raw_bytes, dtype=np.uint8)[:n_rb*4].reshape(4, n_rb).T.flatten().tobytes()
        result = xor_bytes(bt, key)
        if check_flag(result, f'ByteTrans_{direction}+XOR'):
            found = True
            break

if not found:
    print('\n=== Phase 9: RC4 on first 100KB ===')
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
        result = rc4(data, key)
        if check_flag(result, f'RC4_{name}'):
            found = True
            break
        result = rc4(data, key_b64)
        if check_flag(result, f'RC4_{name}_b64key'):
            found = True
            break

if not found:
    print('\n=== Phase 10: Search for ISCC in hex ===')
    for data, name in [(raw_bytes, 'Direct'), (inv_bytes, 'InvTrans')]:
        result = xor_bytes(data, key)
        hx = result.hex()
        idx = hx.find('49534343')
        if idx >= 0:
            print(f'[{name}+XOR] ISCC hex at byte {idx//2}: {result[idx//2:idx//2+30]}')
            found = True

print('\nDone.')
