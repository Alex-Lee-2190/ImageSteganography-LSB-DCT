import os
import random
import hashlib
import struct
import numpy as np
import numba
from functools import lru_cache
from PIL import Image
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

import config
from crypto_utils import kdf_lsb_mode, encrypt_data, decrypt_metadata

# File storage magic header (Distinguishes between plain text and files)
FILE_MAGIC = b'~FILE~'


# ==========================================
# Helper Functions & Numba Kernels
# ==========================================

@lru_cache(maxsize=16)
def _get_shuffled_pixel_pair_map(width, height, location_password):
    """
    Core Security:
    Generates a randomized index map using the location_password.
    Ensures data is physically scattered across pixels based on the key.
    """
    num_pixel_pairs = (width * height) // 2
    # Use password hash/encoding as seed
    rng = np.random.RandomState(seed=list(location_password.encode('utf-8')))
    original_indices = np.arange(num_pixel_pairs, dtype=np.uint32)
    rng.shuffle(original_indices)
    return original_indices


@numba.jit(nopython=True, cache=True)
def _write_bits_kernel_numba(img_array, start_bit_index, bits_to_write_array, shuffled_map):
    height, width, _ = img_array.shape
    total_bits_capacity = len(shuffled_map) * 3

    for i in range(len(bits_to_write_array)):
        bit = bits_to_write_array[i]
        # Logical address: Linear
        logical_bit_addr = start_bit_index + i
        if logical_bit_addr >= total_bits_capacity: break

        logical_pair_index = logical_bit_addr // 3
        channel_index = logical_bit_addr % 3

        # Physical address: Map logical index to randomized pixel pair via shuffled_map
        physical_pair_index = shuffled_map[logical_pair_index]

        y1 = (physical_pair_index * 2) // width
        x1 = (physical_pair_index * 2) % width
        y2 = ((physical_pair_index * 2) + 1) // width
        x2 = ((physical_pair_index * 2) + 1) % width

        if y2 >= height: continue

        p1_channel_val = img_array[y1, x1, channel_index]
        p2_channel_val = img_array[y2, x2, channel_index]
        lsb1 = p1_channel_val & 1
        lsb2 = p2_channel_val & 1

        if (lsb1 ^ lsb2) != bit:
            if i % 2 == 0:
                img_array[y1, x1, channel_index] = p1_channel_val ^ 1
            else:
                img_array[y2, x2, channel_index] = p2_channel_val ^ 1
    return True


@numba.jit(nopython=True, cache=True)
def _read_bits_kernel_numba(img_array, start_bit_index, bit_length, shuffled_map):
    height, width, _ = img_array.shape
    bits = np.empty(bit_length, dtype=np.uint8)
    total_bits_capacity = len(shuffled_map) * 3
    bits_read = 0

    for i in range(bit_length):
        logical_bit_addr = start_bit_index + i
        if logical_bit_addr >= total_bits_capacity: break
        logical_pair_index = logical_bit_addr // 3
        channel_index = logical_bit_addr % 3

        # Reading must also rely on shuffled_map to restore correct physical order
        physical_pair_index = shuffled_map[logical_pair_index]

        y1 = (physical_pair_index * 2) // width
        x1 = (physical_pair_index * 2) % width
        y2 = ((physical_pair_index * 2) + 1) // width
        x2 = ((physical_pair_index * 2) + 1) % width

        if y2 >= height:
            bits[i] = 0
            continue

        p1_val = img_array[y1, x1, channel_index]
        p2_val = img_array[y2, x2, channel_index]
        bits[i] = (p1_val & 1) ^ (p2_val & 1)
        bits_read += 1
    return bits[:bits_read]


def write_bits_to_image(image, start_bit_index, bits_to_write, location_password):
    width, height = image.size
    shuffled_map = _get_shuffled_pixel_pair_map(width, height, location_password)
    bits_array = np.array([int(b) for b in bits_to_write], dtype=np.uint8)
    img_array = np.array(image)
    _write_bits_kernel_numba(img_array, start_bit_index, bits_array, shuffled_map)
    modified_image = Image.fromarray(img_array)
    image.paste(modified_image)
    return True


def read_bits_from_image_mem(image, start_bit_index, bit_length, location_password):
    width, height = image.size
    shuffled_map = _get_shuffled_pixel_pair_map(width, height, location_password)
    img_array = np.array(image)
    bits_array = _read_bits_kernel_numba(img_array, start_bit_index, bit_length, shuffled_map)
    return "".join(map(str, bits_array))


def get_random_start_offset(image, location_password):
    """
    Offset Strategy:
    Limit the start position to the first 5% of the logical stream.
    This prevents capacity issues on large files while maintaining security via shuffling.
    """
    width, height = image.size
    total_bits = (width * height // 2) * 3
    min_required_bits = config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH + config.SECURE_MASTER_BLOCK_BIT_LENGTH

    if total_bits <= min_required_bits: return 0

    # Limit to top 5% of capacity
    limit_zone = total_bits // 20

    # Ensure limit_zone is at least large enough for the header
    if limit_zone < min_required_bits:
        max_start_index = total_bits - min_required_bits
    else:
        max_start_index = limit_zone

    random.seed(location_password.encode('utf-8'))
    start_index = random.randint(0, max_start_index)
    random.seed()
    return start_index


# ==========================================
# Core Functionality: Add & Find (Secure Mode)
# ==========================================

def add_messages_secure(image, messages_to_add, master_password, location_password):
    """
    Embeds messages securely.
    messages_to_add: list of dicts.
      Each dict can contain:
      - 'type': 'text' (default) or 'file'
      - 'key': str (required)
      - 'message': str (if text)
      - 'filename': str (if file)
      - 'data': bytes (if file)
    """
    width, height = image.size
    total_bits_capacity = (width * height // 2) * 3

    base_offset = get_random_start_offset(image, location_password)
    plaintext_header_bin = read_bits_from_image_mem(image, base_offset, config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH,
                                                    location_password)
    is_initialized = len(plaintext_header_bin) == config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH and any(
        b != '0' for b in plaintext_header_bin)

    salt, master_nonce = None, None
    occupied_zones = []

    if is_initialized:
        salt = int(plaintext_header_bin[:config.SECURE_SALT_LEN * 8], 2).to_bytes(config.SECURE_SALT_LEN, 'big')
        master_nonce_start = config.SECURE_SALT_LEN * 8
        master_nonce_end = master_nonce_start + config.SECURE_MASTER_NONCE_LEN * 8
        master_nonce = int(plaintext_header_bin[master_nonce_start:master_nonce_end], 2).to_bytes(
            config.SECURE_MASTER_NONCE_LEN, 'big')
    else:
        salt = os.urandom(config.SECURE_SALT_LEN)
        master_nonce = os.urandom(config.SECURE_MASTER_NONCE_LEN)
        plaintext_header_bytes = salt + master_nonce
        header_bin = ''.join(f'{byte:08b}' for byte in plaintext_header_bytes)
        write_bits_to_image(image, base_offset, header_bin, location_password)

    master_key = kdf_lsb_mode(master_password, salt)
    occupied_zones.append(
        (base_offset, base_offset + config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH + config.SECURE_MASTER_BLOCK_BIT_LENGTH))

    next_block_ptr = 0
    if is_initialized:
        master_block_start = base_offset + config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH
        encrypted_master_block_bin = read_bits_from_image_mem(image, master_block_start,
                                                              config.SECURE_MASTER_BLOCK_BIT_LENGTH, location_password)
        if len(encrypted_master_block_bin) == config.SECURE_MASTER_BLOCK_BIT_LENGTH:
            encrypted_bytes = int(encrypted_master_block_bin, 2).to_bytes(config.SECURE_MASTER_BLOCK_BIT_LENGTH // 8,
                                                                          'big')
            decrypted_bytes = decrypt_metadata(encrypted_bytes, master_key, master_nonce)
            if decrypted_bytes:
                try:
                    magic, head_ptr = struct.unpack('>IQ', decrypted_bytes)
                    if magic == config.SECURE_MAGIC_NUMBER: next_block_ptr = head_ptr
                except struct.error:
                    pass

    last_block_ptr = 0
    current_ptr = next_block_ptr
    while current_ptr != 0:
        current_abs_ptr = base_offset + current_ptr
        encrypted_header_bin = read_bits_from_image_mem(image, current_abs_ptr, config.SECURE_HEADER_BIT_LENGTH,
                                                        location_password)
        if len(encrypted_header_bin) < config.SECURE_HEADER_BIT_LENGTH: return False, "Error: Broken linked list."

        nonce = struct.pack('>Q', current_ptr) + b'\x00' * 4
        decrypted_header_bytes = decrypt_metadata(
            int(encrypted_header_bin, 2).to_bytes(config.SECURE_HEADER_BIT_LENGTH // 8, 'big'), master_key, nonce)
        if not decrypted_header_bytes: return False, "Error: Metadata authentication failed."

        _, _, current_data_len, next_ptr_val = struct.unpack(f'>I{config.SECURE_KEY_HASH_LEN}sIQ',
                                                             decrypted_header_bytes)
        block_size_bits = config.SECURE_HEADER_BIT_LENGTH + (current_data_len * 8) + (config.SECURE_AUTH_TAG_LEN * 8)
        occupied_zones.append((current_abs_ptr, current_abs_ptr + block_size_bits))
        last_block_ptr = current_ptr
        current_ptr = next_ptr_val

    total_to_add = len(messages_to_add)
    for i, item in enumerate(messages_to_add):
        print(f"\rAdding... {i + 1}/{total_to_add}", end="")
        key = item['key']
        key_hash = hashlib.sha256(key.encode('utf-8')).digest()[:config.SECURE_KEY_HASH_LEN]

        # Determine payload type
        payload_bytes = b''
        if 'filename' in item and item['filename']:
            # File Mode: MAGIC + FNAME_LEN + FNAME + DATA
            fname_bytes = item['filename'].encode('utf-8')
            file_data = item['data']
            fname_len = len(fname_bytes)
            if fname_len > 65535: return False, "Error: Filename too long."
            payload_bytes = FILE_MAGIC + struct.pack('>H', fname_len) + fname_bytes + file_data
        else:
            # Text Mode
            message = item['message']
            payload_bytes = message.encode('utf-8')

        data_len = len(payload_bytes)

        new_block_size_bits = config.SECURE_HEADER_BIT_LENGTH + (data_len + config.SECURE_AUTH_TAG_LEN) * 8
        new_block_start_ptr_relative = 0
        max_relative_offset = total_bits_capacity - base_offset - new_block_size_bits

        if max_relative_offset < 0:
            return False, f"Error: Insufficient capacity (Need {new_block_size_bits // 8} bytes)."

        occupied_zones.sort()
        found_spot = False
        for _ in range(1000):
            potential_relative_ptr = random.randint(1, max_relative_offset)
            potential_abs_start = base_offset + potential_relative_ptr
            potential_abs_end = potential_abs_start + new_block_size_bits
            is_overlapping = any(
                max(potential_abs_start, start) < min(potential_abs_end, end) for start, end in occupied_zones)
            if not is_overlapping:
                new_block_start_ptr_relative = potential_relative_ptr
                found_spot = True
                break

        if not found_spot: return False, "Error: Cannot find enough contiguous space (Fragmentation)."

        header_plaintext = struct.pack(f'>I{config.SECURE_KEY_HASH_LEN}sIQ', config.SECURE_MAGIC_NUMBER, key_hash,
                                       data_len, 0)
        header_nonce = struct.pack('>Q', new_block_start_ptr_relative) + b'\x00' * 4
        header_encrypted = encrypt_data(header_plaintext, master_key, header_nonce)

        hkdf_info = b'data-key-for-block-' + struct.pack('>Q', new_block_start_ptr_relative)
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=hkdf_info)
        message_key = hkdf.derive(master_key)
        message_nonce = b'\x01' * 12
        encrypted_message_bytes = encrypt_data(payload_bytes, message_key, message_nonce)

        new_block_bin = ''.join(f'{b:08b}' for b in header_encrypted + encrypted_message_bytes)
        write_bits_to_image(image, base_offset + new_block_start_ptr_relative, new_block_bin, location_password)
        occupied_zones.append((base_offset + new_block_start_ptr_relative,
                               base_offset + new_block_start_ptr_relative + len(new_block_bin)))

        if last_block_ptr == 0:
            master_plaintext = struct.pack('>IQ', config.SECURE_MAGIC_NUMBER, new_block_start_ptr_relative)
            master_encrypted = encrypt_data(master_plaintext, master_key, master_nonce)
            master_bin = ''.join(f'{b:08b}' for b in master_encrypted)
            write_bits_to_image(image, base_offset + config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH, master_bin,
                                    location_password)
        else:
            encrypted_last_header_bin = read_bits_from_image_mem(image, base_offset + last_block_ptr,
                                                                 config.SECURE_HEADER_BIT_LENGTH, location_password)
            nonce = struct.pack('>Q', last_block_ptr) + b'\x00' * 4
            decrypted_last_header = decrypt_metadata(
                int(encrypted_last_header_bin, 2).to_bytes(config.SECURE_HEADER_BIT_LENGTH // 8, 'big'), master_key,
                nonce)
            magic, kh, dl, _ = struct.unpack(f'>I{config.SECURE_KEY_HASH_LEN}sIQ', decrypted_last_header)
            updated_plaintext = struct.pack(f'>I{config.SECURE_KEY_HASH_LEN}sIQ', magic, kh, dl,
                                            new_block_start_ptr_relative)
            updated_encrypted = encrypt_data(updated_plaintext, master_key, nonce)
            updated_header_bin = ''.join(f'{b:08b}' for b in updated_encrypted)
            write_bits_to_image(image, base_offset + last_block_ptr, updated_header_bin, location_password)

        last_block_ptr = new_block_start_ptr_relative

    print()
    return True, "All messages added successfully."


def find_message_secure(image_input, key, master_password, location_password):
    """
    Search for a message.
    Returns: (result_dict, status_msg)
    """
    if isinstance(image_input, str):
        try:
            image = Image.open(image_input).convert("RGBA")
        except FileNotFoundError:
            return None, f"Error: File '{image_input}' not found."
    else:
        image = image_input

    base_offset = get_random_start_offset(image, location_password)
    plaintext_header_bin = read_bits_from_image_mem(image, base_offset, config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH,
                                                    location_password)
    if len(plaintext_header_bin) < config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH: return None, "Error: Incomplete header or wrong location password."

    salt = int(plaintext_header_bin[:config.SECURE_SALT_LEN * 8], 2).to_bytes(config.SECURE_SALT_LEN, 'big')
    master_nonce = int(
        plaintext_header_bin[config.SECURE_SALT_LEN * 8:(config.SECURE_SALT_LEN + config.SECURE_MASTER_NONCE_LEN) * 8],
        2).to_bytes(config.SECURE_MASTER_NONCE_LEN, 'big')

    master_key = kdf_lsb_mode(master_password, salt)
    target_key_hash = hashlib.sha256(key.encode('utf-8')).digest()[:config.SECURE_KEY_HASH_LEN]

    master_block_start = base_offset + config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH
    encrypted_master_bin = read_bits_from_image_mem(image, master_block_start, config.SECURE_MASTER_BLOCK_BIT_LENGTH,
                                                    location_password)
    if len(encrypted_master_bin) < config.SECURE_MASTER_BLOCK_BIT_LENGTH: return None, "Error: Incomplete master block."

    decrypted_master_bytes = decrypt_metadata(
        int(encrypted_master_bin, 2).to_bytes(config.SECURE_MASTER_BLOCK_BIT_LENGTH // 8, 'big'), master_key,
        master_nonce)
    if not decrypted_master_bytes: return None, "Error: Incorrect master password or corrupted block."

    magic, next_block_ptr = struct.unpack('>IQ', decrypted_master_bytes)
    if magic != config.SECURE_MAGIC_NUMBER: return None, "Error: Magic number mismatch."

    while next_block_ptr != 0:
        encrypted_header_bin = read_bits_from_image_mem(image, base_offset + next_block_ptr,
                                                        config.SECURE_HEADER_BIT_LENGTH, location_password)
        if len(encrypted_header_bin) < config.SECURE_HEADER_BIT_LENGTH: return None, "Error: Broken link."

        nonce = struct.pack('>Q', next_block_ptr) + b'\x00' * 4
        decrypted_header_bytes = decrypt_metadata(
            int(encrypted_header_bin, 2).to_bytes(config.SECURE_HEADER_BIT_LENGTH // 8, 'big'), master_key, nonce)

        if not decrypted_header_bytes: return None, "Error: Data corruption."
        _, key_hash_decoded, data_len, next_ptr = struct.unpack(f'>I{config.SECURE_KEY_HASH_LEN}sIQ',
                                                                decrypted_header_bytes)

        if key_hash_decoded == target_key_hash:
            data_start_bit = base_offset + next_block_ptr + config.SECURE_HEADER_BIT_LENGTH
            encrypted_data_len_bits = (data_len * 8) + (config.SECURE_AUTH_TAG_LEN * 8)
            message_bin = read_bits_from_image_mem(image, data_start_bit, encrypted_data_len_bits, location_password)
            encrypted_message_bytes = int(message_bin, 2).to_bytes(data_len + config.SECURE_AUTH_TAG_LEN, 'big')

            hkdf_info = b'data-key-for-block-' + struct.pack('>Q', next_block_ptr)
            hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=hkdf_info)
            message_key = hkdf.derive(master_key)
            message_nonce = b'\x01' * 12

            decrypted_message = decrypt_metadata(encrypted_message_bytes, message_key, message_nonce)
            if not decrypted_message: return None, "Decoding Error: Authentication failed."

            # Check for file format
            if decrypted_message.startswith(FILE_MAGIC):
                try:
                    ptr = len(FILE_MAGIC)
                    fname_len = struct.unpack('>H', decrypted_message[ptr:ptr + 2])[0]
                    ptr += 2
                    filename = decrypted_message[ptr:ptr + fname_len].decode('utf-8')
                    ptr += fname_len
                    file_data = decrypted_message[ptr:]
                    return {'type': 'file', 'filename': filename, 'data': file_data}, "File Found"
                except Exception as e:
                    return None, f"File Parse Error: {e}"
            else:
                try:
                    text_msg = decrypted_message.decode('utf-8')
                    return {'type': 'text', 'message': text_msg}, "Decoded Successfully"
                except:
                    # Compatibility fallback
                    return {'type': 'raw', 'data': decrypted_message}, "Raw Data Found"

        next_block_ptr = next_ptr

    return None, "Error: Message not found."


def find_all_messages(image, master_password, location_password):
    """
    Stress Test: Batch read all messages.
    Returns dict: key_hash -> {'type':..., 'message'/filename/data...}
    """
    all_messages = {}
    base_offset = get_random_start_offset(image, location_password)

    plaintext_header_bin = read_bits_from_image_mem(image, base_offset, config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH,
                                                    location_password)
    if len(plaintext_header_bin) < config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH: return None, "Error: Incomplete header."

    salt = int(plaintext_header_bin[:config.SECURE_SALT_LEN * 8], 2).to_bytes(config.SECURE_SALT_LEN, 'big')
    master_nonce = int(
        plaintext_header_bin[config.SECURE_SALT_LEN * 8:(config.SECURE_SALT_LEN + config.SECURE_MASTER_NONCE_LEN) * 8],
        2).to_bytes(config.SECURE_MASTER_NONCE_LEN, 'big')

    master_key = kdf_lsb_mode(master_password, salt)
    master_block_start = base_offset + config.SECURE_PLAINTEXT_HEADER_BIT_LENGTH
    encrypted_master_bin = read_bits_from_image_mem(image, master_block_start, config.SECURE_MASTER_BLOCK_BIT_LENGTH,
                                                    location_password)

    decrypted_master_bytes = decrypt_metadata(
        int(encrypted_master_bin, 2).to_bytes(config.SECURE_MASTER_BLOCK_BIT_LENGTH // 8, 'big'), master_key,
        master_nonce)
    if not decrypted_master_bytes: return None, "Error: Authentication failed."

    magic, next_block_ptr = struct.unpack('>IQ', decrypted_master_bytes)
    if magic != config.SECURE_MAGIC_NUMBER: return None, "Error: Magic number mismatch."

    while next_block_ptr != 0:
        encrypted_header_bin = read_bits_from_image_mem(image, base_offset + next_block_ptr,
                                                        config.SECURE_HEADER_BIT_LENGTH, location_password)
        nonce = struct.pack('>Q', next_block_ptr) + b'\x00' * 4
        decrypted_header = decrypt_metadata(
            int(encrypted_header_bin, 2).to_bytes(config.SECURE_HEADER_BIT_LENGTH // 8, 'big'), master_key, nonce)

        if not decrypted_header: break

        _, key_hash_decoded, data_len, next_ptr = struct.unpack(f'>I{config.SECURE_KEY_HASH_LEN}sIQ', decrypted_header)

        data_start = base_offset + next_block_ptr + config.SECURE_HEADER_BIT_LENGTH
        enc_data_len = (data_len * 8) + (config.SECURE_AUTH_TAG_LEN * 8)
        msg_bin = read_bits_from_image_mem(image, data_start, enc_data_len, location_password)
        enc_msg_bytes = int(msg_bin, 2).to_bytes(data_len + config.SECURE_AUTH_TAG_LEN, 'big')

        hkdf_info = b'data-key-for-block-' + struct.pack('>Q', next_block_ptr)
        message_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=hkdf_info).derive(master_key)
        decrypted_msg = decrypt_metadata(enc_msg_bytes, message_key, b'\x01' * 12)

        if decrypted_msg:
            try:
                if decrypted_msg.startswith(FILE_MAGIC):
                    ptr = len(FILE_MAGIC)
                    fname_len = struct.unpack('>H', decrypted_msg[ptr:ptr + 2])[0]
                    ptr += 2
                    fn = decrypted_msg[ptr:ptr + fname_len].decode('utf-8')
                    # Stress test needs only meta info
                    all_messages[key_hash_decoded] = {'type': 'file', 'filename': fn}
                else:
                    all_messages[key_hash_decoded] = {'type': 'text', 'message': decrypted_msg.decode('utf-8')}
            except:
                pass

        next_block_ptr = next_ptr

    return all_messages, "Batch Search Complete"