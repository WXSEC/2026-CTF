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
    for pattern in [b'ISCC{', b'iscc{']:
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

n_rows = n // 4

print('\n=== KEY CHECK: Is the data a palindrome? ===')
print('If M = E XOR reverse(NOT(E)), then M should be a palindrome (M[i] = M[N-1-i])')

check_len = min(n, 1000000)
front = arr[:check_len]
back = arr[n-check_len:]
back_rev = np.flip(back)
palindrome_match = np.mean(front == back_rev)
print(f'Raw data palindrome check (first/last {check_len}): {palindrome_match:.6f}')

inv_bits = arr[:n_rows*4].reshape(4, n_rows).T.flatten()
front_inv = inv_bits[:check_len]
back_inv = inv_bits[len(inv_bits)-check_len:]
back_inv_rev = np.flip(back_inv)
palindrome_inv = np.mean(front_inv == back_inv_rev)
print(f'InvTrans data palindrome check (first/last {check_len}): {palindrome_inv:.6f}')

print('\n=== Check palindrome at different positions ===')
for pos_name, pos in [('start', 0), ('quarter', n//4), ('middle', n//2)]:
    front_chunk = arr[pos:pos+1000]
    back_chunk = arr[n-1-pos-999:n-pos]
    back_chunk_rev = np.flip(back_chunk)
    match = np.mean(front_chunk == back_chunk_rev)
    print(f'Position {pos_name} (offset {pos}): palindrome match = {match:.6f}')

print('\n=== Check if data XOR its reverse equals all 1s ===')
front_1m = arr[:1000000]
back_1m = np.flip(arr[n-1000000:])
xor_result = front_1m ^ back_1m
print(f'Mean of (data XOR reversed_data): {np.mean(xor_result):.6f}')
print(f'Fraction of 1s in XOR result: {np.mean(xor_result == 1):.6f}')

print('\n=== Check if NOT(data) equals reversed data ===')
not_data = 1 - arr[:1000000]
rev_data = np.flip(arr[n-1000000:])
print(f'NOT(data_front) == reversed(data_back): {np.mean(not_data == rev_data):.6f}')

print('\n=== Try: data is M = E XOR reverse(NOT(E)), recover E using known plaintext ===')
print('If M[i] = E[i] XOR NOT(E[N-1-i]), then:')
print('  E[i] XOR E[N-1-i] = NOT(M[i]) (since M[i] = E[i] XOR NOT(E[N-1-i]))')
print('  Wait: M[i] = E[i] XOR (1 - E[N-1-i])')
print('  So: E[i] = M[i] XOR (1 - E[N-1-i]) = M[i] XOR NOT(E[N-1-i])')

iscc_prefix = b'ISCC{'
iscc_bits = ''.join(bin(b)[2:].zfill(8) for b in iscc_prefix)
iscc_arr = np.frombuffer(iscc_bits.encode('ascii'), dtype=np.uint8) - ord('0')
print(f'\nISCC{{ in bits: {iscc_bits} ({len(iscc_arr)} bits)')

key_bits = ''.join(bin(b)[2:].zfill(8) for b in key)
key_bit_arr = np.frombuffer(key_bits.encode('ascii'), dtype=np.uint8) - ord('0')

for data_name, data in [('raw', arr), ('inv_trans', inv_bits)]:
    print(f'\n--- Using {data_name} data as M ---')
    N = len(data)

    for start_pos in range(0, min(N, 10000), 8):
        E = np.zeros(N, dtype=np.uint8)
        known = np.zeros(N, dtype=bool)

        for i in range(len(iscc_arr)):
            pos = start_pos + i
            if pos < N:
                E[pos] = iscc_arr[i]
                known[pos] = True

        changed = True
        iterations = 0
        while changed and iterations < 100:
            changed = False
            iterations += 1
            for i in range(N):
                if known[i]:
                    j = N - 1 - i
                    if not known[j]:
                        E[j] = 1 - (data[i] ^ E[i])
                        known[j] = True
                        changed = True

        known_count = np.sum(known)
        if known_count > N // 10:
            E_bytes = bits_to_bytes(E)
            result = xor_key(E_bytes, key)
            if search_flag(result, f'RecoverE_{data_name}_pos{start_pos}'):
                sys.exit(0)

            result2 = E_bytes
            if search_flag(result2, f'RecoverE_noXOR_{data_name}_pos{start_pos}'):
                sys.exit(0)

            if start_pos < 100:
                show_first(result, f'RecoverE_{data_name}_pos{start_pos}+XOR', 30)

    print(f'  Checked positions 0 to {min(N, 10000)}')

print('\n=== Alternative: maybe the encoding is simpler ===')
print('What if truth.dat = D XOR key (at bit level), and D is the flag padded?')

for data_name, data in [('raw', arr), ('inv_trans', inv_bits)]:
    data_bytes = bits_to_bytes(data)
    for key_name, key_bytes in [('decoded', key), ('b64', key_b64)]:
        result = xor_key(data_bytes, key_bytes)
        text = result.decode('utf-8', errors='replace')
        idx = text.find('ISCC{')
        if idx >= 0:
            end = text.find('}', idx)
            if end >= 0:
                flag = text[idx:end+1]
                print(f'*** FLAG FOUND [{data_name}+{key_name}]: {flag} ***')
                sys.exit(0)

        result_lsb = bits_to_bytes_lsb(data)
        result_lsb = xor_key(result_lsb, key_bytes)
        text = result_lsb.decode('utf-8', errors='replace')
        idx = text.find('ISCC{')
        if idx >= 0:
            end = text.find('}', idx)
            if end >= 0:
                flag = text[idx:end+1]
                print(f'*** FLAG FOUND [{data_name}+LSB+{key_name}]: {flag} ***')
                sys.exit(0)

print('\n=== Try: data represents a Möbius strip that needs to be "unrolled" ===')
print('The strip has width 4 (四位成组), and the twist connects left to right with flip')

mat4 = arr[:n_rows*4].reshape(n_rows, 4)
print(f'Matrix shape: {mat4.shape}')

print('\nCheck: does column 0 relate to column 3 (and col 1 to col 2) via Möbius?')
for c1, c2 in [(0, 3), (1, 2)]:
    col1 = mat4[:, c1]
    col2 = mat4[:, c2]
    col2_rev = np.flip(col2)
    col2_rev_flip = 1 - col2_rev
    match = np.mean(col1 == col2_rev_flip)
    match2 = np.mean(col1 == col2_rev)
    match3 = np.mean(col1 == col2)
    print(f'  Col{c1} vs Col{c2}: direct={match3:.6f}, reversed={match2:.6f}, rev+flip={match:.6f}')

print('\nCheck: does top of each column relate to bottom via Möbius?')
half_rows = n_rows // 2
for c in range(4):
    col = mat4[:, c]
    top = col[:half_rows]
    bottom = col[half_rows:]
    bottom_rev_flip = np.flip(1 - bottom)
    match = np.mean(top == bottom_rev_flip)
    match2 = np.mean(top == np.flip(bottom))
    match3 = np.mean(top == 1 - bottom)
    print(f'  Col{c}: top==rev_flip(bottom)={match:.6f}, top==rev(bottom)={match2:.6f}, top==flip(bottom)={match3:.6f}')

print('\n=== Try: unroll Möbius strip by reading row by row with twist ===')
unrolled = []
for r in range(n_rows):
    row = mat4[r].copy()
    unrolled.append(row)
unrolled = np.array(unrolled)

for flip_cols in [True, False]:
    for reverse_rows in [True, False]:
        result_mat = unrolled.copy()
        if reverse_rows:
            result_mat = np.flip(result_mat, axis=0)
        if flip_cols:
            result_mat = 1 - result_mat

        combined = np.vstack([unrolled, result_mat])
        combined_bytes = bits_to_bytes(combined.flatten())
        for key_name, key_bytes in [('decoded', key), ('b64', key_b64), ('none', b'')]:
            if key_bytes:
                result = xor_key(combined_bytes, key_bytes)
            else:
                result = combined_bytes
            if search_flag(result, f'Unroll_flip{flip_cols}_rev{reverse_rows}+{key_name}'):
                sys.exit(0)

print('\n=== Try: the flag might be encoded differently - check for base64 ===')
for data_name, data in [('raw', arr), ('inv_trans', inv_bits)]:
    data_bytes = bits_to_bytes(data)
    for key_name, key_bytes in [('decoded', key), ('b64', key_b64), ('none', b'')]:
        if key_bytes:
            xored = xor_key(data_bytes, key_bytes)
        else:
            xored = data_bytes
        try:
            decoded = base64.b64decode(xored[:1000], validate=False)
            if search_flag(decoded, f'B64_{data_name}+{key_name}'):
                sys.exit(0)
        except:
            pass

print('\nDone with all approaches. No flag found yet.')
