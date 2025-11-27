import time
import random
import string
import os
import shutil
import hashlib
import sys
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project Modules
import lsb_stego
import freq_stego
import config


# ==========================================
# Utility: Suppress Output
# ==========================================
class SuppressOutput:
    """Context manager to suppress stdout in subprocesses."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ==========================================
# Attacks (For FFT Robustness)
# ==========================================
def attack_graffiti_scribble(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Stroke width relative to image size
        min_dim = min(w, h)
        min_width = max(1, int(min_dim * 0.01))
        max_width = max(2, int(min_dim * 0.03))

        for _ in range(15):
            points = [(random.randint(0, w), random.randint(0, h)) for _ in range(4)]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            width = random.randint(min_width, max_width)
            draw.line(points, fill=color, width=width, joint="curve")
        img.save(save_path)
    except:
        pass


def attack_crop_random_40(img_path, save_path):
    try:
        img = Image.open(img_path)
        w, h = img.size
        target_w = int(w * 0.4)
        target_h = int(h * 0.4)
        if w > target_w and h > target_h:
            left = random.randint(0, w - target_w)
            top = random.randint(0, h - target_h)
        else:
            left, top = 0, 0
        crop_img = img.crop((left, top, left + target_w, top + target_h))
        crop_img.save(save_path)
    except:
        pass


def attack_rotation_90(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.rotate(90, expand=True).save(save_path)
    except:
        pass


def attack_rotation_180(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.rotate(180).save(save_path)
    except:
        pass


def attack_jpeg_50(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.save(save_path, "JPEG", quality=50)
    except:
        pass


def attack_salt_pepper_noise(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        rows, cols, _ = arr.shape
        mask = np.random.random((rows, cols))
        arr[mask < 0.025] = [0, 0, 0]
        arr[mask > 0.975] = [255, 255, 255]
        Image.fromarray(arr).save(save_path)
    except:
        pass


def attack_gaussian_noise(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32)
        # Mean 0, Sigma 25
        noise = np.random.normal(0, 25, arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(save_path)
    except:
        pass


def attack_blur(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.filter(ImageFilter.GaussianBlur(radius=1.5)).save(save_path)
    except:
        pass


def attack_multi_occlusion(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Block size relative
        min_dim = min(w, h)
        block_size = max(10, int(min_dim * 0.1))

        for _ in range(20):
            max_x = max(0, w - block_size)
            max_y = max(0, h - block_size)

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            draw.rectangle([(x, y), (x + block_size, y + block_size)], fill=(0, 0, 0))
        img.save(save_path)
    except:
        pass


def attack_brightness(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        enhancer = ImageEnhance.Brightness(img)
        # Increase brightness by 50%
        img = enhancer.enhance(1.5)
        img.save(save_path)
    except:
        pass


def attack_contrast(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        enhancer = ImageEnhance.Contrast(img)
        # Increase contrast by 50%
        img = enhancer.enhance(1.5)
        img.save(save_path)
    except:
        pass


def attack_scaling_50(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # Scale down to 50% then back up (Lossy interpolation)
        img = img.resize((w // 2, h // 2), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        img.save(save_path)
    except:
        pass


ATTACK_MAP = {
    "Scribble": attack_graffiti_scribble,
    "Crop (40% Area)": attack_crop_random_40,
    "Rotate (90)": attack_rotation_90,
    "Rotate (180)": attack_rotation_180,
    "Occlusion (20 Blocks)": attack_multi_occlusion,
    "JPEG (Q50)": attack_jpeg_50,
    "Noise (Salt&Pepper)": attack_salt_pepper_noise,
    "Noise (Gaussian)": attack_gaussian_noise,
    "Blur (R1.5)": attack_blur,
    "Brightness (+50%)": attack_brightness,
    "Contrast (+50%)": attack_contrast,
    "Scaling (50%)": attack_scaling_50
}


# ==========================================
# Test Logic
# ==========================================

def run_lsb_stress_test(base_image_path, count):
    """
    Stress test for LSB Secure File System Mode.
    """
    print(f"\n--- Start Stress Test (LSB Secure): {count} messages ---")
    try:
        image = Image.open(base_image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Base image '{base_image_path}' not found.")
        return

    test_master_password = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    test_location_password = ''.join(random.choices(string.ascii_letters + string.digits, k=32))

    test_data = []
    for _ in range(count):
        msg_len = random.randint(20, 50)
        key_len = random.randint(10, 20)
        test_data.append({
            'type': 'text',
            'message': ''.join(random.choices(string.ascii_letters + string.digits, k=msg_len)),
            'key': ''.join(random.choices(string.ascii_letters + string.digits, k=key_len))
        })

    print("Adding messages in memory...")
    start_time = time.time()
    success, msg = lsb_stego.add_messages_secure(image, test_data, test_master_password, test_location_password)
    if not success:
        print(f"Failed: {msg}")
        return
    print(f"Added. Time: {time.time() - start_time:.2f} sec.")

    if not os.path.exists("image"):
        os.makedirs("image")
    final_image_path = os.path.join("image", "stress_test_lsb_temp.png")

    try:
        image.save(final_image_path, "PNG")
    except Exception as e:
        print(f"Save temp file failed: {e}")
        return

    print(f"Verifying (Temp file: {final_image_path})...")
    verify_start = time.time()
    try:
        image_verify = Image.open(final_image_path).convert("RGBA")
        decoded_map, status = lsb_stego.find_all_messages(image_verify, test_master_password,
                                                                     test_location_password)

        success_count = 0
        if decoded_map:
            for item in test_data:
                key_hash = hashlib.sha256(item['key'].encode('utf-8')).digest()[:config.SECURE_KEY_HASH_LEN]
                if key_hash in decoded_map:
                     found_item = decoded_map[key_hash]
                     if found_item.get('type') == 'text' and found_item.get('message') == item['message']:
                        success_count += 1

        print(f"Verify Complete. Success: {success_count}/{count}. Time: {time.time() - verify_start:.2f}s")
    finally:
        # Cleanup
        if os.path.exists(final_image_path):
            try:
                os.remove(final_image_path)
            except:
                pass


def _process_single_image_fft(args):
    """Subprocess for FFT batch test"""
    idx, total, img_path, output_dir, content, password, mode = args
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]

    local_stats = {name: 0 for name in ATTACK_MAP.keys()}

    with SuppressOutput():
        stego_path = os.path.join(output_dir, f"{base_name}_stego.png")

        try:
            success, msg = freq_stego.encrypt_and_embed_dct(
                img_path, content, password, stego_path
            )
            if not success:
                return (base_name, False, f"Embed Fail: {msg}", local_stats)
        except Exception as e:
            return (base_name, False, f"Embed Error: {e}", local_stats)

        for attack_name, attack_func in ATTACK_MAP.items():
            attack_filename = f"{base_name}_{attack_name.split(' ')[0]}_attack.png"
            if "JPEG" in attack_name:
                attack_filename = attack_filename.replace(".png", ".jpg")

            attacked_path = os.path.join(output_dir, attack_filename)

            try:
                attack_func(stego_path, attacked_path)
                res_data, res_type, _ = freq_stego.decrypt_and_extract_fft(attacked_path, password)

                if res_data:
                    # Verification
                    if res_data == content:
                        local_stats[attack_name] = 1

            except:
                pass

    return (base_name, True, "OK", local_stats)


def run_fft_robustness_test(input_dir=os.path.join("image", "test_images"), mode='text', max_workers=8):
    """
    Batch Robustness Test for FFT.
    """
    OUTPUT_DIR = os.path.join("image", "batch_results")
    PASSWORD = "Alex_Lee"

    if mode == 'text':
        TEST_CONTENT = "Google LLC is an American technology corporation."
    elif mode == 'qrcode':
        TEST_CONTENT = "Google.com"
    else:
        print("Unsupported mode")
        return

    # Check directory
    if not os.path.exists(input_dir):
        try:
            os.makedirs(input_dir)
            print(f"[!] Input dir '{input_dir}' not found. Created. Please add images.")
        except:
            print(f"[!] Input dir '{input_dir}' not found and cannot be created.")
        return

    # Cleanup/Create output dir
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
        except:
            pass
    os.makedirs(OUTPUT_DIR)

    # Collect images
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"[!] No images in '{input_dir}'.")
        return

    print("=" * 60)
    print(f"  FFT Batch Robustness Tool")
    print(f"  Mode: {mode.upper()}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Workers: {max_workers}")
    print(f"  Images: {len(image_files)}")
    print("=" * 60)

    global_stats = {name: {'success': 0, 'total': 0} for name in ATTACK_MAP.keys()}

    tasks = []
    for i, f in enumerate(image_files):
        tasks.append((i, len(image_files), f, OUTPUT_DIR, TEST_CONTENT, PASSWORD, mode))

    start_time = time.time()
    completed_count = 0
    failed_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_image_fft, t): t for t in tasks}

        print(f"Processing...")

        for future in as_completed(futures):
            completed_count += 1
            base_name, success, msg, result_stats = future.result()

            percent = (completed_count / len(tasks)) * 100

            if success:
                print(f"\r[{completed_count}/{len(tasks)}] {percent:.1f}% - {base_name:<15} [OK]", end="")
                for k, v in result_stats.items():
                    global_stats[k]['total'] += 1
                    global_stats[k]['success'] += v
            else:
                print(f"\n❌ Failed [{base_name}]: {msg}")
                failed_list.append(base_name)

    print("\n" + "=" * 60)
    print(f"  Done! Time: {time.time() - start_time:.2f}s")
    if failed_list:
        print(f"  [Warning] {len(failed_list)} images failed.")
    print("=" * 60)
    print(f"{'Attack Type':<25} | {'Pass Rate':<10} | {'S/T'}")
    print("-" * 60)

    overall_succ = 0
    overall_tot = 0

    for name, data in global_stats.items():
        s = data['success']
        t = data['total']
        rate = (s / t) * 100 if t > 0 else 0
        print(f"{name:<25} | {rate:6.1f}%    | {s}/{t}")
        overall_succ += s
        overall_tot += t

    print("-" * 60)
    if overall_tot > 0:
        total_rate = (overall_succ / overall_tot) * 100
        print(f"{'Overall Score':<25} | {total_rate:6.1f}%    | {overall_succ}/{overall_tot}")
    print("=" * 60)