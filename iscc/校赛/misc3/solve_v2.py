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
    for pattern in [b'ISCC{', b'iscc{', b'Iscc{']:
        idx2 = data.find(pattern)
        if idx2 >= 0:
            end = data.find(b'}', idx2)
            if end >= 0:
                flag = data[idx2:end+1].decode('utf-8', errors='replace')
                print(f'*** FLAG FOUND [{label}]: {flag} ***')
                return True
    return False

def show_first(data, label, n=80):
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

first_half = arr[:half]
second_half = arr[half:2*half]

print('\n=== Verify Möbius relationships at different levels ===')

raw_bytes = bits_to_bytes(arr)
raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
n_bytes = len(raw_arr)
byte_first = raw_arr[:n_bytes//2]
byte_second = raw_arr[n_bytes//2:]

print(f'Bit level: first == flip(reverse(second)): {np.mean(first_half == np.flip(1-second_half)):.6f}')
print(f'Bit level: first == reverse(flip(second)): {np.mean(first_half == 1-np.flip(second_half)):.6f}')
print(f'Byte level: first == flip(reverse(second)): {np.mean(byte_first == np.flip(255-byte_second)):.6f}')
print(f'Byte level: first == reverse(flip(second)): {np.mean(byte_first == 255-np.flip(byte_second)):.6f}')

n_nibbles = n // 4
nibble_mat = arr[:n_nibbles*4].reshape(n_nibbles, 4)
nib_first = nibble_mat[:n_nibbles//2]
nib_second = nibble_mat[n_nibbles//2:]
nib_flip_rev = np.flip(1 - nib_second, axis=0)
print(f'Nibble level: first == flip(reverse(second)): {np.mean(nib_first == nib_flip_rev):.6f}')

print('\n=== Key insight: "起点亦或终点" with Möbius twist ===')
print('The encoding: original D -> Möbius strip -> D + reverse(flip(D))')
print('Then XOR with key, then transpose')
print('Decoding: inverse transpose -> XOR with key -> take first half')
print('OR: inverse transpose -> take first half -> XOR with key')

inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
inv_bytes = bits_to_bytes(inv_bits)

print('\n--- Check: after inverse transpose, does Möbius relationship hold? ---')
inv_first = inv_bits[:len(inv_bits)//2]
inv_second = inv_bits[len(inv_bits)//2:]
print(f'InvTrans bit: first == flip(reverse(second)): {np.mean(inv_first == np.flip(1-inv_second)):.6f}')

inv_byte_arr = np.frombuffer(inv_bytes, dtype=np.uint8)
inv_byte_first = inv_byte_arr[:len(inv_byte_arr)//2]
inv_byte_second = inv_byte_arr[len(inv_byte_arr)//2:]
print(f'InvTrans byte: first == flip(reverse(second)): {np.mean(inv_byte_first == np.flip(255-inv_byte_second)):.6f}')

print('\n=== Comprehensive approach: try ALL operation orderings ===')
operations = []

for trans_name, trans_fn in [
    ('no_trans', lambda d: d),
    ('inv_trans_4', lambda d: d[:n_rows*4].reshape(4, n_rows).T.flatten()),
]:
    for mobius_name, mobius_fn in [
        ('no_mobius', lambda d: d),
        ('xor_rev_flip', lambda d: d[:len(d)//2] ^ np.flip(1-d[len(d)//2:])),
        ('take_first_half', lambda d: d[:len(d)//2]),
    ]:
        for xor_name, xor_fn in [
            ('no_xor', lambda d: d),
            ('xor_key', lambda d: xor_key(d, key)),
            ('xor_b64key', lambda d: xor_key(d, key_b64)),
        ]:
            for byte_name, byte_fn in [
                ('msb', bits_to_bytes),
                ('lsb', bits_to_bytes_lsb),
            ]:
                ops_label = f'{trans_name}+{mobius_name}+{xor_name}+{byte_name}'
                operations.append((ops_label, trans_fn, mobius_fn, xor_fn, byte_fn))

found = False
for ops_label, trans_fn, mobius_fn, xor_fn, byte_fn in operations:
    if found:
        break
    try:
        data = arr.copy()
        data = trans_fn(data)
        data = mobius_fn(data)
        data_bytes = byte_fn(data)
        data_bytes = xor_fn(data_bytes)
        if search_flag(data_bytes, ops_label):
            found = True
            break
    except Exception as e:
        pass

if not found:
    print('\nNo flag found with basic operation orderings.')
    print('Trying XOR before transposition...')

for trans_name, trans_fn in [
    ('no_trans', lambda d: d),
    ('inv_trans_4', lambda d: d[:n_rows*4].reshape(4, n_rows).T.flatten()),
]:
    if found:
        break
    for xor_name, xor_fn in [
        ('xor_key', lambda d: xor_key(d, key)),
        ('xor_b64key', lambda d: xor_key(d, key_b64)),
    ]:
        if found:
            break
        for mobius_name, mobius_fn in [
            ('no_mobius', lambda d: d),
            ('xor_rev_flip', lambda d: d[:len(d)//2] ^ np.flip(1-d[len(d)//2:])),
            ('take_first_half', lambda d: d[:len(d)//2]),
        ]:
            if found:
                break
            for byte_name, byte_fn in [
                ('msb', bits_to_bytes),
                ('lsb', bits_to_bytes_lsb),
            ]:
                ops_label = f'xor_first_{xor_name}+{trans_name}+{mobius_name}+{byte_name}'
                try:
                    data_bytes = raw_bytes.copy()
                    data_bytes = xor_fn(data_bytes)
                    xor_arr = np.frombuffer(data_bytes, dtype=np.uint8)
                    xor_bits = np.unpackbits(xor_arr)
                    data = xor_bits[:n]
                    data = trans_fn(data)
                    data = mobius_fn(data)
                    data_bytes = byte_fn(data)
                    if search_flag(data_bytes, ops_label):
                        found = True
                        break
                except Exception as e:
                    pass

if not found:
    print('\nNo flag found with XOR-before-transpose orderings either.')
    print('Trying RC4 and other approaches...')

for data_name, data_bytes in [('raw', raw_bytes), ('inv_trans', inv_bytes)]:
    if found:
        break
    for key_name, key_bytes in [('decoded', key), ('b64', key_b64)]:
        if found:
            break
        rc4_result = rc4_decrypt(data_bytes, key_bytes)
        if search_flag(rc4_result, f'RC4_{data_name}_{key_name}'):
            found = True
            break
        for mobius_name in ['take_first_half']:
            half_bytes = len(rc4_result) // 2
            first_half_rc4 = rc4_result[:half_bytes]
            if search_flag(first_half_rc4, f'RC4_{data_name}_{key_name}_{mobius_name}'):
                found = True
                break

if not found:
    print('\nNo flag found with RC4 either.')
    print('Trying byte-level Möbius + transposition...')

    for width in [4, 2, 8, 16, 21]:
        if n_bytes % width != 0:
            continue
        rows_b = n_bytes // width
        mat_b = raw_arr[:rows_b*width].reshape(rows_b, width)

        byte_first_h = mat_b[:rows_b//2]
        byte_second_h = mat_b[rows_b//2:]

        for mob_type, mob_fn in [
            ('xor_rev_flip', lambda t,b: t ^ np.flip(255-b, axis=0)),
            ('xor_rev_flip_lr', lambda t,b: t ^ np.flip(np.flip(255-b, axis=1), axis=0)),
            ('take_first', lambda t,b: t),
        ]:
            untwisted = mob_fn(byte_first_h, byte_second_h)
            untwisted_bytes = untwisted.flatten().tobytes()
            for key_name, key_bytes in [('decoded', key), ('b64', key_b64)]:
                result = xor_key(untwisted_bytes, key_bytes)
                if search_flag(result, f'ByteMobius_w{width}_{mob_type}+XOR_{key_name}'):
                    found = True
                    break
                rc4_r = rc4_decrypt(untwisted_bytes, key_bytes)
                if search_flag(rc4_r, f'ByteMobius_w{width}_{mob_type}+RC4_{key_name}'):
                    found = True
                    break
            if found:
                break
        if found:
            break

if not found:
    print('\nNo flag found with byte-level Möbius either.')
    print('Trying known-plaintext attack with ISCC{ prefix...')

    iscc_bytes = b'ISCC{'
    iscc_arr = np.frombuffer(iscc_bytes, dtype=np.uint8)

    for data_name, data_bytes in [('raw', raw_bytes), ('inv_trans', inv_bytes)]:
        if found:
            break
        data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
        for key_name, key_bytes in [('decoded', key), ('b64', key_b64)]:
            if found:
                break
            key_arr = np.frombuffer(key_bytes, dtype=np.uint8)
            key_len = len(key_arr)
            for offset in range(min(key_len, 21)):
                expected = np.bitwise_xor(iscc_arr, key_arr[offset:offset+5])
                matches = 0
                for pos in range(0, len(data_arr) - 5, key_len):
                    if np.array_equal(data_arr[pos:pos+5], expected):
                        matches += 1
                        full_start = pos
                        remaining = data_arr[full_start:]
                        decrypted = xor_key(remaining.tobytes(), key_bytes)
                        text = decrypted.decode('utf-8', errors='replace')
                        end = text.find('}', 5)
                        if end >= 0 and end < 200:
                            flag = text[:end+1]
                            if 'ISCC{' in flag:
                                print(f'*** FLAG FOUND [KPA_{data_name}_{key_name}_off{offset}]: {flag} ***')
                                found = True
                                break
                if matches > 0:
                    print(f'[KPA_{data_name}_{key_name}_off{offset}] {matches} matches found')
            if found:
                break

if not found:
    print('\nStill no flag. Trying more creative approaches...')

    print('\n=== Try: bit-level XOR with key, THEN inverse transpose, THEN Möbius ===')
    key_bits = ''.join(bin(b)[2:].zfill(8) for b in key)
    key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')
    key_tiled = np.tile(key_bit_arr, len(arr) // len(key_bit_arr) + 1)[:len(arr)]
    xored = arr ^ key_tiled

    xored_inv = xored[:n_rows*4].reshape(4, n_rows).T.flatten()
    xored_inv_bytes = bits_to_bytes(xored_inv)
    if search_flag(xored_inv_bytes, 'BitXOR_InvTrans'):
        found = True

    xored_inv_first = xored_inv[:len(xored_inv)//2]
    xored_inv_first_bytes = bits_to_bytes(xored_inv_first)
    if not found and search_flag(xored_inv_first_bytes, 'BitXOR_InvTrans_FirstHalf'):
        found = True

    print('\n=== Try: transpose at different widths after Möbius ===')
    mobius_result = first_half ^ np.flip(1 - second_half)
    for width in [2, 4, 8, 16, 21, 32, 64]:
        m_len = len(mobius_result)
        if m_len % width != 0:
            continue
        m_rows = m_len // width
        m_mat = mobius_result[:m_rows*width].reshape(m_rows, width)
        m_inv = m_mat.T.flatten()
        m_bytes = bits_to_bytes(m_inv)
        result = xor_key(m_bytes, key)
        if search_flag(result, f'Mobius_InvTrans_w{width}+XOR'):
            found = True
            break

if not found:
    print('\n=== Last resort: search for partial patterns ===')
    for data_name, data_bytes in [('raw', raw_bytes), ('inv_trans', inv_bytes)]:
        data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
        for key_name, key_bytes in [('decoded', key), ('b64', key_b64)]:
            result = xor_key(data_bytes, key_bytes)
            result_arr = np.frombuffer(result, dtype=np.uint8)
            printable = np.array([32 <= b < 127 for b in result_arr])
            max_run = 0
            max_pos = 0
            current_run = 0
            for i in range(len(printable)):
                if printable[i]:
                    current_run += 1
                    if current_run > max_run:
                        max_run = current_run
                        max_pos = i - current_run + 1
                else:
                    current_run = 0
            if max_run >= 10:
                print(f'[{data_name}+{key_name}] Longest printable run: {max_run} at pos {max_pos}')
                print(f'  Content: {result[max_pos:max_pos+max_run]}')

print('\nDone!')
