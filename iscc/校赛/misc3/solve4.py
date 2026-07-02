import base64
import numpy as np
import sys
import io
from itertools import permutations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

key_b64 = b'WXRoOVVyMDYyYXpaQTA5eTRyczVM'
key = base64.b64decode(key_b64)
print(f'Key: {key}, len: {len(key)}')

with open('truth.dat', 'r') as f:
    truth = f.read().strip()

n = len(truth)
arr = np.frombuffer(truth.encode('ascii'), dtype=np.uint8) - ord('0')
del truth

n_rows = n // 4
seg = n_rows

def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    rem = len(bits) % 8
    if rem:
        bits = bits[:len(bits) - rem]
    bits = bits.reshape(-1, 8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()

def xor_key_bytes(data, key_bytes):
    key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    key_tiled = np.tile(key_arr, len(data_arr) // len(key_arr) + 1)[:len(data_arr)]
    return (data_arr ^ key_tiled).tobytes()

def xor_key_bits(bits_arr, key_bytes):
    key_bits = ''.join(bin(b)[2:].zfill(8) for b in key_bytes)
    key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
    key_tiled = np.tile(key_bit_arr, len(bits_arr) // len(key_bit_arr) + 1)[:len(bits_arr)]
    return bits_arr ^ key_tiled

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
    return False

print('=== Try all 24 column permutations x 16 reversal patterns ===')
segments = [arr[i*seg:(i+1)*seg] for i in range(4)]

found = False
count = 0
for perm in permutations(range(4)):
    if found:
        break
    for rev_mask in range(16):
        count += 1
        cols = []
        for i in range(4):
            col = segments[perm[i]].copy()
            if rev_mask & (1 << i):
                col = np.flip(col)
            cols.append(col)
        
        mat = np.column_stack(cols)
        bits = mat.flatten()
        bytes_data = bits_to_bytes(bits)
        result = xor_key_bytes(bytes_data, key)
        
        text = result[:200].decode('utf-8', errors='replace')
        idx = text.find('ISCC{')
        if idx >= 0:
            end = text.find('}', idx)
            if end >= 0:
                flag = text[idx:end+1]
                print(f'*** FLAG FOUND! perm={perm}, rev={rev_mask:#06b} ***')
                print(f'Flag: {flag}')
                found = True
                break
            print(f'Partial: perm={perm}, rev={rev_mask:#06b}, pos={idx}: {repr(text[idx:idx+50])}')
        
        if count % 50 == 0:
            print(f'  Checked {count}/384 combinations...')

print(f'Total checked: {count}')

if not found:
    print('\nTrying bit-XOR first, then column permutations...')
    xored_arr = xor_key_bits(arr, key)
    xored_segs = [xored_arr[i*seg:(i+1)*seg] for i in range(4)]
    
    for perm in permutations(range(4)):
        if found:
            break
        for rev_mask in range(16):
            cols = []
            for i in range(4):
                col = xored_segs[perm[i]].copy()
                if rev_mask & (1 << i):
                    col = np.flip(col)
                cols.append(col)
            
            mat = np.column_stack(cols)
            bits = mat.flatten()
            bytes_data = bits_to_bytes(bits)
            
            text = bytes_data[:200].decode('utf-8', errors='replace')
            idx = text.find('ISCC{')
            if idx >= 0:
                end = text.find('}', idx)
                if end >= 0:
                    flag = text[idx:end+1]
                    print(f'*** FLAG FOUND! BitXOR, perm={perm}, rev={rev_mask:#06b} ***')
                    print(f'Flag: {flag}')
                    found = True
                    break

if not found:
    print('\nTrying Möbius untwist + inverse transposition...')
    half = n // 2
    first = arr[:half]
    second = arr[half:]
    
    for flip in [True, False]:
        if found:
            break
        for rev in [True, False]:
            mod_second = second.copy()
            if flip:
                mod_second = 1 - mod_second
            if rev:
                mod_second = np.flip(mod_second)
            
            combined = np.concatenate([first, mod_second])
            c_rows = len(combined) // 4
            
            inv_c = combined[:c_rows*4].reshape(4, c_rows).T.flatten()
            inv_c_bytes = bits_to_bytes(inv_c)
            result = xor_key_bytes(inv_c_bytes, key)
            label = f'Mobius_f={flip}_r={rev}+InvTrans+XOR'
            if search_flag(result[:500], label):
                found = True
                break

if not found:
    print('\nTrying different key formats with inverse transposition...')
    inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
    inv_bytes = bits_to_bytes(inv_bits)
    
    for k, kname in [(key, 'decoded'), (key_b64, 'b64'), (b'Yth9Ur062azZA09y4rs5L', 'string')]:
        result = xor_key_bytes(inv_bytes, k)
        if search_flag(result[:500], f'InvTrans+XOR_{kname}'):
            found = True
            break

if not found:
    print('\nTrying LSB byte order with all approaches...')
    def bits_to_bytes_lsb(bits):
        bits = np.asarray(bits, dtype=np.uint8)
        rem = len(bits) % 8
        if rem:
            bits = bits[:len(bits) - rem]
        bits = bits.reshape(-1, 8)
        weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
        return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()
    
    inv_lsb = bits_to_bytes_lsb(inv_bits)
    result = xor_key_bytes(inv_lsb, key)
    if search_flag(result[:500], 'InvTrans_LSB+XOR'):
        found = True
    
    if not found:
        direct_lsb = bits_to_bytes_lsb(arr)
        result = xor_key_bytes(direct_lsb, key)
        if search_flag(result[:500], 'Direct_LSB+XOR'):
            found = True

if not found:
    print('\n=== Searching full data for ISCC pattern ===')
    raw_bytes = bits_to_bytes(arr)
    result = xor_key_bytes(raw_bytes, key)
    print('Searching Direct+XOR full data...')
    search_flag(result, 'Direct+XOR_full')
    
    result = xor_key_bytes(inv_bytes, key)
    print('Searching InvTrans+XOR full data...')
    search_flag(result, 'InvTrans+XOR_full')

print('\nDone.')
