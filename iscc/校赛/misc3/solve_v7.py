import base64
import numpy as np
import sys
import io
import hashlib

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

def xor_key_bytes(data, key_bytes):
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

print('\n=== Check Möbius property AFTER inverse transposition + XOR ===')
inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes(inv_bits)

for key_name, key_bytes in [('dec', key), ('b64', key_b64),
                             ('md5_dec', hashlib.md5(key).digest()),
                             ('sha256_dec', hashlib.sha256(key).digest()[:16])]:
    result = xor_key_bytes(inv_bytes, key_bytes)
    result_arr = np.frombuffer(result, dtype=np.uint8)
    n_b = len(result_arr)
    byte_first = result_arr[:n_b//2]
    byte_second = result_arr[n_b//2:]
    
    match_rev_flip = np.mean(byte_first == np.flip(255-byte_second))
    match_rev = np.mean(byte_first == np.flip(byte_second))
    match_flip = np.mean(byte_first == 255-byte_second)
    
    print(f'{key_name}: rev_flip={match_rev_flip:.6f}, rev={match_rev:.6f}, flip={match_flip:.6f}')

print('\n=== Check Möbius property at BIT level after inverse transposition + XOR ===')
for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
    key_bits = ''.join(bin(b)[2:].zfill(8) for b in key_bytes)
    key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
    key_tiled = np.tile(key_bit_arr, len(inv_bits) // len(key_bit_arr) + 1)[:len(inv_bits)]
    xored_inv = inv_bits ^ key_tiled
    
    xfirst = xored_inv[:half]
    xsecond = xored_inv[half:2*half]
    
    match_rev_flip = np.mean(xfirst == np.flip(1-xsecond))
    match_rev = np.mean(xfirst == np.flip(xsecond))
    match_flip = np.mean(xfirst == (1-xsecond))
    
    print(f'BitXOR {key_name}: rev_flip={match_rev_flip:.6f}, rev={match_rev:.6f}, flip={match_flip:.6f}')

print('\n=== Try: XOR with key, then check 4-col matrix Möbius ===')
for key_name, key_bytes in [('dec', key), ('b64', key_b64)]:
    key_bits = ''.join(bin(b)[2:].zfill(8) for b in key_bytes)
    key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
    key_tiled = np.tile(key_bit_arr, len(arr) // len(key_bit_arr) + 1)[:len(arr)]
    xored = arr ^ key_tiled
    
    xmat4 = xored[:n_rows*4].reshape(n_rows, 4)
    half_r = n_rows // 2
    xtop = xmat4[:half_r]
    xbottom = xmat4[half_r:2*half_r]
    
    match_both = np.mean(xtop == np.flip(np.flip(1-xbottom, axis=1), axis=0))
    match_ud = np.mean(xtop == np.flip(1-xbottom, axis=0))
    match_lr = np.mean(xtop == np.flip(1-xbottom, axis=1))
    
    print(f'Matrix BitXOR {key_name}: both_flip={match_both:.6f}, ud_flip={match_ud:.6f}, lr_flip={match_lr:.6f}')

print('\n=== Try: the encoding might use a different bit order within nibbles ===')
inv_nibbles = inv_bits[:len(inv_bits)//4*4].reshape(-1, 4)

for bit_order_name, bit_order in [
    ('reverse', [3,2,1,0]),
    ('swap01', [1,0,2,3]),
    ('swap23', [0,1,3,2]),
    ('rotate1', [1,2,3,0]),
    ('rotate2', [2,3,0,1]),
    ('rotate3', [3,0,1,2]),
]:
    reordered = inv_nibbles[:, bit_order]
    reordered_bytes = bits_to_bytes(reordered.flatten())
    result = xor_key_bytes(reordered_bytes, key)
    if search_flag(result, f'BitOrder_{bit_order_name}+XOR'):
        sys.exit(0)

print('\n=== Try: bit reversal within each BYTE ===')
inv_byte_arr = np.frombuffer(inv_bytes, dtype=np.uint8)
reversed_bits_per_byte = np.unpackbits(inv_byte_arr)[:, ::-1]  # This doesn't work directly
# Let me do it properly
inv_bits_reshaped = inv_bits[:len(inv_bits)//8*8].reshape(-1, 8)
reversed_bytes = inv_bits_reshaped[:, ::-1]
rev_bytes = bits_to_bytes(reversed_bytes.flatten())
result = xor_key_bytes(rev_bytes, key)
if search_flag(result, 'BitRevPerByte+XOR'):
    sys.exit(0)

print('\n=== Try: swap nibbles within each byte ===')
inv_nibbles_for_bytes = inv_bits[:len(inv_bits)//8*8].reshape(-1, 8)
swapped = inv_nibbles_for_bytes[:, [4,5,6,7,0,1,2,3]]
swapped_bytes = bits_to_bytes(swapped.flatten())
result = xor_key_bytes(swapped_bytes, key)
if search_flag(result, 'NibbleSwap+XOR'):
    sys.exit(0)

print('\n=== Try: key-derived starting position ===')
key_sum = sum(key)
key_b64_sum = sum(key_b64)
print(f'Key sum: {key_sum}, b64 key sum: {key_b64_sum}')

for start_pos in [key_sum, key_b64_sum, 0, 1, 8, 21, 168, 336]:
    if start_pos >= len(inv_bytes):
        continue
    result = xor_key_bytes(inv_bytes[start_pos:], key)
    if search_flag(result, f'InvTrans_start{start_pos}+XOR'):
        sys.exit(0)

print('\n=== Try: use Chinese text from secret.dat as key ===')
chinese_text = "取一个长方形纸带将其末端翻转与首端粘合后可以在现实世界中得到莫比乌斯环"
chinese_key = chinese_text.encode('utf-8')
result = xor_key_bytes(inv_bytes[:100000], chinese_key)
if search_flag(result, 'ChineseKey+XOR'):
    sys.exit(0)
printable = sum(1 for b in result[:100] if 32 <= b < 127)
print(f'Chinese key: printable ratio = {printable/100:.2f}')

poem_text = "四位成组拆骨分藏纵向拾取各归其行零壹铺路字符浮光四言成谶水落石方"
poem_key = poem_text.encode('utf-8')
result = xor_key_bytes(inv_bytes[:100000], poem_key)
if search_flag(result, 'PoemKey+XOR'):
    sys.exit(0)
printable = sum(1 for b in result[:100] if 32 <= b < 127)
print(f'Poem key: printable ratio = {printable/100:.2f}')

print('\n=== Try: double XOR (XOR with key, then XOR with b64 key) ===')
result = xor_key_bytes(inv_bytes, key)
result = xor_key_bytes(result, key_b64)
if search_flag(result, 'DoubleXOR'):
    sys.exit(0)

print('\n=== Try: the data might be a simple substitution of the flag ===')
print('Check if specific bit patterns repeat with the key period')
key_bit_len = len(key) * 8  # 168 bits
sample_len = key_bit_len * 100
sample = arr[:sample_len]
chunks = sample.reshape(100, key_bit_len)
mean_per_bit = np.mean(chunks, axis=0)
std_per_bit = np.std(chunks, axis=0)
print(f'Mean of bit positions across key periods: min={mean_per_bit.min():.4f}, max={mean_per_bit.max():.4f}')
print(f'Std of bit positions: min={std_per_bit.min():.4f}, max={std_per_bit.max():.4f}')
biased_bits = np.where(np.abs(mean_per_bit - 0.5) > 0.02)[0]
print(f'Biased bit positions (|mean-0.5|>0.02): {len(biased_bits)} out of {key_bit_len}')

print('\n=== Try: treat data as encoded with 4b/5b or similar ===')
print('360469472 / 5 =', 360469472 / 5, '(not integer)')
print('360469472 / 10 =', 360469472 / 10, '(not integer)')

print('\n=== Last resort: try every possible 2-operation combination ===')
transforms = {
    'raw': arr,
    'inv4': arr[:n_rows*4].reshape(4, n_rows).T.flatten(),
    'inv8': arr[:n//8*8].reshape(8, n//8).T.flatten(),
}

mobius_ops = {
    'none': lambda d: d,
    'first_half': lambda d: d[:len(d)//2],
    'xor_rev_flip': lambda d: d[:len(d)//2] ^ np.flip(1-d[len(d)//2:len(d)//2*2]),
}

for t_name, t_data in transforms.items():
    for m_name, m_fn in mobius_ops.items():
        data = m_fn(t_data)
        data_bytes = bits_to_bytes(data)
        for k_name, k_bytes in [('dec', key), ('b64', key_b64), ('none', b'')]:
            if k_bytes:
                result = xor_key_bytes(data_bytes, k_bytes)
            else:
                result = data_bytes
            if search_flag(result, f'{t_name}+{m_name}+{k_name}'):
                sys.exit(0)

print('\nAll approaches exhausted. No flag found.')
