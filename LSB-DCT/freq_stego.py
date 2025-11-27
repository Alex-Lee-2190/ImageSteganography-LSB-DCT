import os
import struct
import random
import math
import time
import zlib
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2
import qrcode
from reedsolo import RSCodec, ReedSolomonError
from crypto_utils import kdf_fft_mode, encrypt_data, decrypt_metadata

# Try import PyTorch
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

# ==============================================================================
#  Configuration: Robustness Settings
# ==============================================================================
BLOCK_SIZE = 8

# Use classic asymmetric coefficients for better spatial separation
C1 = (3, 3)
C2 = (2, 4)

# Strength Configuration
BASE_STRENGTH = 30.0
MAX_STRENGTH = 90.0
LOG_FACTOR = 10.0

# Luminance Masking
LUM_MASK_FACTOR = 0.5

# Sync Beacon
SYNC_RADIUS_RATIO = 0.35
SYNC_TARGET_ANGLE = 40.0
BASE_SYNC_STRENGTH = 100000.0

TILE_H_BLOCKS = 36
TILE_W_BLOCKS = 36
TILE_CAPACITY_BITS = TILE_H_BLOCKS * TILE_W_BLOCKS
TILE_CAPACITY_BYTES = TILE_CAPACITY_BITS // 8

MAGIC_BYTES = b'Tiny'
MAGIC_BITS_LEN = 32

# Compact Header
SALT_RANDOM_LEN = 8
HEADER_ECC_SYMBOLS = 13
HEADER_DATA_LEN = (len(MAGIC_BYTES) + SALT_RANDOM_LEN) + 1 + 1
HEADER_TOTAL_LEN = HEADER_DATA_LEN + HEADER_ECC_SYMBOLS
HEADER_BITS_LEN = HEADER_TOTAL_LEN * 8

MAX_BODY_TOTAL_BYTES = TILE_CAPACITY_BYTES - HEADER_TOTAL_LEN

TYPE_TEXT = 0
TYPE_LEGACY_QR = 2


# ==========================================
# Utils
# ==========================================
def generate_qr_image(text_content):
    try:
        qr = qrcode.QRCode(version=3, box_size=10, border=4, error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(text_content)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        return img
    except Exception as e:
        print(f"QR Gen Failed: {e}")
        return None


def _get_permutation_map(length, seed):
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices


def _scramble_bits(bits_str, seed):
    length = len(bits_str)
    perm = _get_permutation_map(length, seed)
    scrambled = ['0'] * length
    for i, target_idx in enumerate(perm):
        scrambled[target_idx] = bits_str[i]
    return "".join(scrambled)


def _unscramble_bits(bits_str, seed):
    length = len(bits_str)
    perm = _get_permutation_map(length, seed)
    unscrambled = ['0'] * length
    for i, target_idx in enumerate(perm):
        unscrambled[i] = bits_str[target_idx]
    return "".join(unscrambled)


def _format_result(payload, mode_val):
    try:
        decompressed_bytes = zlib.decompress(payload)
    except:
        decompressed_bytes = payload
    try:
        text_content = decompressed_bytes.decode('utf-8', errors='ignore')
    except:
        text_content = str(decompressed_bytes)
    return text_content, 'text', "Extraction Success"


# ==============================================================================
#  Part 1: CPU Encoder
# ==============================================================================
def _dct_block_cpu(block):
    return cv2.dct(block.astype(np.float32))


def _idct_block_cpu(block):
    return cv2.idct(block)


def _add_fft_template_cpu(img_arr):
    Y = img_arr[:, :, 0].astype(np.float32)
    h, w = Y.shape

    global_mean = np.mean(Y)
    safe_mean = max(global_mean, 30.0)
    adaptive_sync_strength = BASE_SYNC_STRENGTH * (safe_mean / 128.0)

    f = np.fft.fft2(Y)
    fshift = np.fft.fftshift(f)
    radius = min(h, w) * SYNC_RADIUS_RATIO
    cy, cx = h // 2, w // 2
    angle_rad = math.radians(SYNC_TARGET_ANGLE)
    off_y = int(radius * math.sin(angle_rad))
    off_x = int(radius * math.cos(angle_rad))

    for sign in [1, -1]:
        p = (cy - sign * off_y, cx + sign * off_x)
        if 0 <= p[0] < h and 0 <= p[1] < w:
            val = fshift[p]
            mag = np.abs(val)
            ang = np.angle(val)
            fshift[p] = (mag + adaptive_sync_strength) * np.exp(1j * ang)

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    img_arr[:, :, 0] = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_arr


def encrypt_and_embed_dct(image_path, content, password, output_path):
    try:
        img = Image.open(image_path).convert('YCbCr')
    except Exception as e:
        return False, f"Open Failed: {e}"

    print(f"  [Info] Injecting frequency beacon...")
    img_arr = np.array(img)
    img_arr = _add_fft_template_cpu(img_arr)
    Y = img_arr[:, :, 0].astype(np.float32)

    h_blocks = Y.shape[0] // BLOCK_SIZE
    w_blocks = Y.shape[1] // BLOCK_SIZE
    if h_blocks < TILE_H_BLOCKS or w_blocks < TILE_W_BLOCKS:
        return False, f"Image too small ({h_blocks}x{w_blocks}) < ({TILE_H_BLOCKS}x{TILE_W_BLOCKS})."

    raw_bytes = content.encode('utf-8')
    try:
        compressed = zlib.compress(raw_bytes, level=9)
        payload_bytes = compressed if len(compressed) < len(raw_bytes) else raw_bytes
    except:
        payload_bytes = raw_bytes

    mode_val = TYPE_TEXT
    salt = MAGIC_BYTES + os.urandom(SALT_RANDOM_LEN)
    key, nonce = kdf_fft_mode(password, salt)
    encrypted_body = encrypt_data(payload_bytes, key, nonce)

    body_len = len(encrypted_body)
    available_ecc_bytes = MAX_BODY_TOTAL_BYTES - body_len
    if available_ecc_bytes < 4:
        return False, f"Data too long (Exceeds by {abs(available_ecc_bytes)} B)."

    header_raw = salt + struct.pack('>B', mode_val) + struct.pack('>B', len(encrypted_body))
    rs_header = RSCodec(HEADER_ECC_SYMBOLS)
    try:
        header_encoded = rs_header.encode(header_raw)
        rs_body = RSCodec(available_ecc_bytes)
        body_encoded = rs_body.encode(encrypted_body)
    except Exception as e:
        return False, f"ECC Error: {e}"

    full_bits = ''.join(f'{b:08b}' for b in header_encoded) + ''.join(f'{b:08b}' for b in body_encoded)

    rng = random.Random(password)
    padding_len = TILE_CAPACITY_BITS - len(full_bits)
    padding_bits = "".join([str(rng.randint(0, 1)) for _ in range(padding_len)])
    final_bits = full_bits + padding_bits
    scrambled_bits = _scramble_bits(final_bits, seed=password)

    temp_indices = [(r, c) for r in range(TILE_H_BLOCKS) for c in range(TILE_W_BLOCKS)]
    random.seed(password)
    random.shuffle(temp_indices)
    coord_to_idx = np.zeros((TILE_H_BLOCKS, TILE_W_BLOCKS), dtype=np.int32)
    for i, (rr, rc) in enumerate(temp_indices):
        coord_to_idx[rr, rc] = i

    for r in range(h_blocks):
        for c in range(w_blocks):
            rel_r = r % TILE_H_BLOCKS
            rel_c = c % TILE_W_BLOCKS
            bit_idx = coord_to_idx[rel_r, rel_c]
            bit = scrambled_bits[bit_idx]

            y_start, x_start = r * BLOCK_SIZE, c * BLOCK_SIZE
            block = Y[y_start: y_start + BLOCK_SIZE, x_start: x_start + BLOCK_SIZE]

            # --- HVS Calculation ---
            block_std = np.std(block)
            block_mean = np.mean(block)

            # Texture enhancement
            texture_strength = BASE_STRENGTH + LOG_FACTOR * math.log(1.0 + block_std)
            # Luminance masking
            lum_dist = abs(block_mean - 128.0) / 128.0
            lum_factor = 1.0 + (LUM_MASK_FACTOR * (lum_dist ** 2))

            final_strength = texture_strength * lum_factor
            final_strength = min(final_strength, MAX_STRENGTH)
            final_strength = max(final_strength, BASE_STRENGTH)

            block_centered = block - 128.0
            dct_b = _dct_block_cpu(block_centered)
            v1 = dct_b[C1]
            v2 = dct_b[C2]

            if bit == '1':
                if v1 <= v2 + final_strength:
                    diff = (v2 + final_strength - v1) / 2.0 + 1.0
                    v1 += diff
                    v2 -= diff
            else:
                if v2 <= v1 + final_strength:
                    diff = (v1 + final_strength - v2) / 2.0 + 1.0
                    v2 += diff
                    v1 -= diff

            dct_b[C1] = v1
            dct_b[C2] = v2

            # Reconstruct
            block_recon = _idct_block_cpu(dct_b) + 128.0

            # --- DC Offset (Anti-Clipping) ---
            b_min, b_max = block_recon.min(), block_recon.max()
            shift = 0.0

            if b_max > 255.0 and b_min >= (b_max - 255.0):
                shift = -(b_max - 255.0)
            elif b_min < 0.0 and b_max <= (255.0 + b_min):
                shift = -b_min
            elif b_max > 255.0 and b_min < 0.0:
                center = (b_max + b_min) / 2.0
                shift = 127.5 - center

            block_recon += shift

            # Write back
            Y[y_start: y_start + BLOCK_SIZE, x_start: x_start + BLOCK_SIZE] = block_recon

    img_arr[:, :, 0] = np.clip(Y, 0, 255).astype(np.uint8)
    try:
        Image.fromarray(img_arr, mode='YCbCr').convert('RGB').save(output_path)
        return True, f"Steganography Success\nSaved to: {output_path}"
    except Exception as e:
        return False, f"Save Failed: {e}"


# ==============================================================================
#  Part 2: GPU Decoder
# ==============================================================================

def _generate_dct_kernel_gpu():
    N = 8

    def get_basis(u, v):
        basis = torch.zeros((N, N))
        alpha_u = math.sqrt(1 / N) if u == 0 else math.sqrt(2 / N)
        alpha_v = math.sqrt(1 / N) if v == 0 else math.sqrt(2 / N)
        for x in range(N):
            for y in range(N):
                basis[x, y] = alpha_u * alpha_v * \
                              math.cos((2 * x + 1) * u * math.pi / (2 * N)) * \
                              math.cos((2 * y + 1) * v * math.pi / (2 * N))
        return basis

    basis_c1 = get_basis(C1[0], C1[1])
    basis_c2 = get_basis(C2[0], C2[1])
    kernel = (basis_c1 - basis_c2).unsqueeze(0).unsqueeze(0)
    return kernel.to(DEVICE)


def _get_gpu_coord_map(password):
    temp_indices = [(r, c) for r in range(TILE_H_BLOCKS) for c in range(TILE_W_BLOCKS)]
    random.seed(password)
    random.shuffle(temp_indices)
    coord_map_np = np.zeros((TILE_H_BLOCKS, TILE_W_BLOCKS), dtype=np.int64)
    for i, (rr, rc) in enumerate(temp_indices):
        coord_map_np[rr, rc] = i
    return torch.from_numpy(coord_map_np).to(DEVICE)


def _precompute_magic_indices(password):
    temp_indices = [(r, c) for r in range(TILE_H_BLOCKS) for c in range(TILE_W_BLOCKS)]
    random.seed(password)
    random.shuffle(temp_indices)
    perm = _get_permutation_map(TILE_CAPACITY_BITS, password)
    magic_coords = []
    for i in range(MAGIC_BITS_LEN):
        idx_in_scrambled = perm[i]
        r, c = temp_indices[idx_in_scrambled]
        magic_coords.append([r, c])
    return torch.tensor(magic_coords, device=DEVICE, dtype=torch.long)


def _gpu_detect_rotation(img_tensor):
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    fft = torch.fft.fft2(img_tensor)
    fshift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fshift).squeeze()
    cy, cx = H // 2, W // 2
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing='ij')
    dist_sq = (y_grid - cy) ** 2 + (x_grid - cx) ** 2
    center_r = min(H, W) * 0.1
    magnitude[dist_sq <= center_r ** 2] = 0
    ignore_w = 5
    magnitude[cy - ignore_w:cy + ignore_w, :] = 0
    magnitude[:, cx - ignore_w:cx + ignore_w] = 0
    target_r = min(H, W) * SYNC_RADIUS_RATIO
    bw = 15
    ring_mask = (dist_sq > (target_r - bw) ** 2) & (dist_sq < (target_r + bw) ** 2)
    masked_mag = magnitude * ring_mask
    peak_idx = torch.argmax(masked_mag)
    peak_val = masked_mag.view(-1)[peak_idx]
    if peak_val < 5000: return None
    peak_y = peak_idx // W
    peak_x = peak_idx % W
    dy = cy - peak_y.item()
    dx = peak_x.item() - cx
    detected_deg = math.degrees(math.atan2(dy, dx))
    if detected_deg < 0: detected_deg += 180
    diff = detected_deg - SYNC_TARGET_ANGLE
    if diff > 90: diff -= 180
    if diff < -90: diff += 180
    return diff


def _gpu_cyclic_vote_and_decode(diff_map, coord_map_gpu, magic_coords_gpu, password):
    rs_header = RSCodec(HEADER_ECC_SYMBOLS)

    shifts_y = torch.arange(TILE_H_BLOCKS, device=DEVICE)
    shifts_x = torch.arange(TILE_W_BLOCKS, device=DEVICE)
    grid_shifts_y, grid_shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    grid_shifts_y = grid_shifts_y.reshape(-1)
    grid_shifts_x = grid_shifts_x.reshape(-1)

    magic_bits_str = ''.join(f'{b:08b}' for b in MAGIC_BYTES)
    magic_target = torch.tensor([int(b) for b in magic_bits_str], device=DEVICE, dtype=torch.float32)

    phases = []
    phases.append((0, 0))
    for py in range(8):
        for px in range(8):
            if py == 0 and px == 0: continue
            phases.append((py, px))

    for (py, px) in phases:
        grid_raw = diff_map[0, 0, py::8, px::8]
        grid = torch.sign(grid_raw)

        gh, gw = grid.shape
        if gh < 10 or gw < 10: continue

        pad_h = (TILE_H_BLOCKS - gh % TILE_H_BLOCKS) % TILE_H_BLOCKS
        pad_w = (TILE_W_BLOCKS - gw % TILE_W_BLOCKS) % TILE_W_BLOCKS
        if pad_h > 0 or pad_w > 0:
            grid = F.pad(grid, (0, pad_w, 0, pad_h))

        grid_folded = grid.view(-1, TILE_H_BLOCKS, grid.shape[1]).sum(dim=0)
        grid_folded = grid_folded.view(TILE_H_BLOCKS, -1, TILE_W_BLOCKS).sum(dim=1)

        row_indices = (magic_coords_gpu[:, 0].unsqueeze(0) + grid_shifts_y.unsqueeze(1)) % TILE_H_BLOCKS
        col_indices = (magic_coords_gpu[:, 1].unsqueeze(0) + grid_shifts_x.unsqueeze(1)) % TILE_W_BLOCKS

        extracted_magic = grid_folded[row_indices, col_indices]
        extracted_bits = (extracted_magic > 0).float()
        matches = (extracted_bits == magic_target.unsqueeze(0)).sum(dim=1)

        best_match_idx = torch.argmax(matches)
        best_match_score = matches[best_match_idx].item()

        if best_match_score < 28:
            continue

        best_dy = grid_shifts_y[best_match_idx]
        best_dx = grid_shifts_x[best_match_idx]

        grid_shifted = torch.roll(grid_folded, shifts=(-int(best_dy), -int(best_dx)), dims=(0, 1))

        votes_flat = grid_shifted.view(-1)
        indices_flat = coord_map_gpu.view(-1)
        final_votes = torch.zeros(TILE_CAPACITY_BITS, device=DEVICE)
        final_votes.scatter_add_(0, indices_flat, votes_flat)

        bits_np = (final_votes > 0).cpu().numpy().astype(np.int8)
        bits_str = "".join(bits_np.astype(str))

        raw_bits = _unscramble_bits(bits_str, seed=password)

        header_bits = raw_bits[:HEADER_BITS_LEN]
        try:
            h_bytes = bytes([int(header_bits[i:i + 8], 2) for i in range(0, len(header_bits), 8)])
            dec_header = rs_header.decode(h_bytes)[0]
            dec_header = bytes(dec_header)

            salt = dec_header[:12]  # Magic(4) + Random(8)
            mode_val = dec_header[12]
            body_len = struct.unpack('>B', dec_header[13:14])[0]

            expected_ecc_len = MAX_BODY_TOTAL_BYTES - body_len
            if expected_ecc_len < 4: continue

            body_total_len = body_len + expected_ecc_len
            body_bits_len = body_total_len * 8

            if HEADER_BITS_LEN + body_bits_len > len(raw_bits): continue

            body_bits = raw_bits[HEADER_BITS_LEN: HEADER_BITS_LEN + body_bits_len]
            b_bytes = bytes([int(body_bits[i:i + 8], 2) for i in range(0, len(body_bits), 8)])

            rs_body = RSCodec(expected_ecc_len)
            dec_body = rs_body.decode(b_bytes)[0]
            dec_body = bytes(dec_body)

            key, nonce = kdf_fft_mode(password, salt)
            payload = decrypt_metadata(dec_body, key, nonce)

            if payload:
                return payload, mode_val
        except:
            continue

    return None, None


def decrypt_and_extract_fft(image_path, password):
    if not TORCH_AVAILABLE:
        return None, None, "Error: PyTorch not installed."

    try:
        img_pil = Image.open(image_path).convert('RGB')
    except:
        return None, None, "Open Failed"

    dct_kernel = _generate_dct_kernel_gpu()
    coord_map_gpu = _get_gpu_coord_map(password)
    magic_coords_gpu = _precompute_magic_indices(password)

    img_tensor = TF.to_tensor(img_pil).to(DEVICE) * 255.0
    Y_tensor = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
    Y_tensor = Y_tensor.unsqueeze(0).unsqueeze(0) - 128.0

    print(f"  [GPU] Decoding on {DEVICE}...")

    # Strategy A: 0 Degree
    diff_map = F.conv2d(Y_tensor, dct_kernel, stride=1)
    res, mode = _gpu_cyclic_vote_and_decode(diff_map, coord_map_gpu, magic_coords_gpu, password)
    if res:
        print(f"    [Success] 0 deg Hit!")
        return _format_result(res, mode)

    # Strategy B: Rotation
    rot_offset = _gpu_detect_rotation(Y_tensor)
    if rot_offset is not None:
        if abs(rot_offset) > 0.5:
            print(f"    [Correction] Rotating {-rot_offset:.2f} deg")
            Y_rot = TF.rotate(Y_tensor, -rot_offset, interpolation=TF.InterpolationMode.BILINEAR, expand=True,
                              fill=-128)
            diff_map = F.conv2d(Y_rot, dct_kernel, stride=1)
            res, mode = _gpu_cyclic_vote_and_decode(diff_map, coord_map_gpu, magic_coords_gpu, password)
            if res: return _format_result(res, mode)

            # Flip 180
            rot_180 = -rot_offset + 180
            Y_rot = TF.rotate(Y_tensor, rot_180, interpolation=TF.InterpolationMode.BILINEAR, expand=True, fill=-128)
            diff_map = F.conv2d(Y_rot, dct_kernel, stride=1)
            res, mode = _gpu_cyclic_vote_and_decode(diff_map, coord_map_gpu, magic_coords_gpu, password)
            if res: return _format_result(res, mode)
        else:
            Y_rot = TF.rotate(Y_tensor, 180, interpolation=TF.InterpolationMode.BILINEAR, expand=True, fill=-128)
            diff_map = F.conv2d(Y_rot, dct_kernel, stride=1)
            res, mode = _gpu_cyclic_vote_and_decode(diff_map, coord_map_gpu, magic_coords_gpu, password)
            if res: return _format_result(res, mode)

    # Strategy C: Brute Force
    angles = [90, 270, -1, 1, -2, 2]
    for ang in angles:
        Y_rot = TF.rotate(Y_tensor, ang, interpolation=TF.InterpolationMode.BILINEAR, expand=True, fill=-128)
        diff_map = F.conv2d(Y_rot, dct_kernel, stride=1)
        res, mode = _gpu_cyclic_vote_and_decode(diff_map, coord_map_gpu, magic_coords_gpu, password)
        if res: return _format_result(res, mode)

    return None, None, "Extraction Failed"