import os
import sys
import time
from PIL import Image

# Import preserved modules
import lsb_stego
import freq_stego
import tests
import config

# --- Config: Default credentials ---
DEFAULT_OPEN_MASTER_PW = "PUBLIC_MASTER_PASSWORD_DEFAULT"
DEFAULT_OPEN_LOC_PW = "PUBLIC_LOCATION_PASSWORD_DEFAULT"


def main():
    while True:
        print("\n" + "=" * 60)
        print(" Image Steganography System ")
        print("=" * 60)
        print("\n [A] Spatial Domain ")
        print(" (1) Add Data")
        print(" (2) Find Data")
        print("\n [B] Frequency Domain ")
        print(" (3) Embed Watermark")
        print(" (4) Extract Watermark")
        print("\n [C] Stress Test ")
        print(" (5) Spatial Domain / Capacity")
        print(" (6) Frequency Domain / Robustness")
        print("\n (q) Quit")

        choice = input("\nSelect Option -> ")

        # ==============================================================================
        # Feature 1: LSB - Add Data (Text/File)
        # ==============================================================================
        if choice == '1':
            print("\n--- Spatial Domain: Add Data ---")
            input_image_path = input("Enter source image path: ")
            try:
                image_in_memory = Image.open(input_image_path).convert("RGBA")
            except Exception as e:
                print(f"Error: Cannot open image. {e}")
                continue

            print("\n[1] Add Text Message")
            print("[2] Add File")
            sub_choice = input("Select Type [1/2]: ")

            use_encryption = input("Enable Password Encryption? (y/n) [Default: y]: ").lower().strip() != 'n'

            master_pw = ""
            location_pw = ""

            if use_encryption:
                print(">> Mode: [Encrypted Storage]")
                master_pw = input("Enter [Password] (for encryption and distribution): ")
                location_pw = master_pw
                if not master_pw:
                    print("Error: Password cannot be empty.")
                    continue
            else:
                print(">> Mode: [Public Storage] (Using default credentials)")
                master_pw = DEFAULT_OPEN_MASTER_PW
                location_pw = DEFAULT_OPEN_LOC_PW

            messages_to_add = []

            if sub_choice == '1':
                # Add text
                try:
                    num_messages = int(input("How many messages to add? "))
                except ValueError:
                    print("Invalid number.")
                    continue

                for i in range(num_messages):
                    print(f"--- Message #{i + 1} ---")
                    message = input(f"Content: ")
                    key_prompt = "Key" if use_encryption else "Label"
                    key = input(f"Enter {key_prompt} (Lookup ID): ")
                    if not key:
                        print("Error: Key/Label cannot be empty.")
                        break
                    messages_to_add.append({'type': 'text', 'message': message, 'key': key})

            elif sub_choice == '2':
                # Add file
                file_path = input("Enter file path to hide: ")
                if not os.path.exists(file_path):
                    print("File does not exist.")
                    continue
                try:
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    fname = os.path.basename(file_path)
                    print(f"File loaded: {fname} ({len(file_data)} bytes)")

                    key_prompt = "Extraction Key" if use_encryption else "Label"
                    key = input(f"Enter {key_prompt}: ")
                    if not key:
                        print("Error: Key cannot be empty.")
                        continue
                    messages_to_add.append({'type': 'file', 'filename': fname, 'data': file_data, 'key': key})
                except Exception as e:
                    print(f"Read file failed: {e}")
                    continue
            else:
                print("Invalid selection.")
                continue

            if messages_to_add:
                start_time = time.time()
                success, msg = lsb_stego.add_messages_secure(image_in_memory, messages_to_add, master_pw,
                                                                 location_pw)

                if success:
                    path_parts = os.path.splitext(input_image_path)
                    suffix = "_secure" if use_encryption else "_open"
                    default_output = f"{path_parts[0]}{suffix}.png"
                    output_path = input(f"Save as (Enter for '{default_output}'): ") or default_output
                    try:
                        image_in_memory.save(output_path, "PNG")
                        print(f"\nSuccess! Time: {time.time() - start_time:.2f}s. Saved to '{output_path}'")
                    except Exception as e:
                        print(f"Save failed: {e}")
                else:
                    print(f"\nFailed: {msg}")

        # ==============================================================================
        # Feature 2: LSB - Find Data
        # ==============================================================================
        elif choice == '2':
            print("\n--- Spatial Domain: Find Data ---")
            path = input("Enter image path: ")
            if not os.path.exists(path):
                print("Error: File not found.")
                continue

            is_encrypted = input("Is the file encrypted? (y/n) [Default: y]: ").lower().strip() != 'n'

            master_pw = ""
            location_pw = ""

            if is_encrypted:
                master_pw = input("Enter [Password]: ")
                location_pw = master_pw
            else:
                print(">> Using default public credentials...")
                master_pw = DEFAULT_OPEN_MASTER_PW
                location_pw = DEFAULT_OPEN_LOC_PW

            key_prompt = "Key" if is_encrypted else "Label"
            key = input(f"Enter search {key_prompt}: ")

            res_dict, msg = lsb_stego.find_message_secure(path, key, master_pw, location_pw)

            if res_dict:
                print(f"\n[{msg}]")
                if res_dict.get('type') == 'file':
                    fname = res_dict['filename']
                    fsize = len(res_dict['data'])
                    print(f"File Detected: {fname} ({fsize} bytes)")
                    save_path = input(f"Save path (Default '{fname}'): ") or fname
                    try:
                        with open(save_path, 'wb') as f:
                            f.write(res_dict['data'])
                        print(f"File saved to: {save_path}")
                    except Exception as e:
                        print(f"Save failed: {e}")

                elif res_dict.get('type') == 'text':
                    print(f"Content: \n{res_dict['message']}")

                else:
                    print(f"Raw Data: {res_dict.get('data')}")
            else:
                print(f"\n[Search Failed] {msg}")

        # ==============================================================================
        # Feature 3: FFT - Embed Watermark
        # ==============================================================================
        elif choice == '3':
            print("\n--- FFT: Embed Watermark ---")
            data_content = input("Enter watermark content (Text/URL): ")
            if not data_content:
                print("Content cannot be empty.")
                continue

            path = input("Enter carrier image path: ")
            if not os.path.exists(path):
                print("Error: File not found.")
                continue

            use_encryption = input("Enable Encryption? (y/n) [Default: y]: ").lower().strip() != 'n'

            password = ""
            if use_encryption:
                password = input("Enter password: ")
                if not password:
                    print("Error: Password cannot be empty.")
                    continue
            else:
                print(">> Mode: [Public Watermark] (Default Seed)")
                password = DEFAULT_OPEN_MASTER_PW

            path_parts = os.path.splitext(path)
            default_output = f"{path_parts[0]}_fft.png"
            output_path = input(f"Save as (Enter for '{default_output}'): ") or default_output

            print("Processing...")
            success, txt = freq_stego.encrypt_and_embed_dct(path, data_content, password, output_path)
            print(txt)

        # ==============================================================================
        # Feature 4: FFT - Extract Watermark
        # ==============================================================================
        elif choice == '4':
            print("\n--- FFT: Extract Watermark ---")
            path = input("Enter image path: ")
            if not os.path.exists(path):
                print("Error: File not found.")
                continue

            is_encrypted = input("Is watermark encrypted? (y/n) [Default: y]: ").lower().strip() != 'n'

            password = ""
            if is_encrypted:
                password = input("Enter password: ")
            else:
                print(">> Using default public credentials...")
                password = DEFAULT_OPEN_MASTER_PW

            print("Extracting...")
            res_data, res_type, msg = freq_stego.decrypt_and_extract_fft(path, password)

            if res_data:
                print(f"\n[{msg}]")
                print(f"Content: {res_data}")

                ask_qr = input("\nGenerate QR Code? (y/n) [Default: n]: ").lower().strip()
                if ask_qr == 'y':
                    img = freq_stego.generate_qr_image(res_data)
                    if img:
                        default_save = "extracted_content_qr.png"
                        save_path = input(f"Save QR as (Default '{default_save}'): ") or default_save
                        try:
                            img.save(save_path)
                            print(f"QR saved to: {save_path}")
                        except Exception as e:
                            print(f"Save failed: {e}")
                    else:
                        print("QR generation failed.")
            else:
                print(f"\nExtraction Failed: {msg}")

        # ==============================================================================
        # Feature 5: Stress Test (LSB)
        # ==============================================================================
        elif choice == '5':
            print("\n--- LSB Stress Test ---")
            path = input("Enter [Original] image path: ")
            if not os.path.exists(path):
                print("Error: File not found.")
                continue
            try:
                count = int(input("Number of messages (e.g. 100): "))
                tests.run_lsb_stress_test(path, count)
            except ValueError:
                print("Invalid number.")

        # ==============================================================================
        # Feature 6: Robustness Test (FFT)
        # ==============================================================================
        elif choice == '6':
            print("\n--- FFT Robustness Test ---")
            default_dir = os.path.join("image", "test_images")
            input_dir = input(f"Enter image folder path (Default '{default_dir}'): ") or default_dir

            print(" [1] Text Mode")
            print(" [2] QR Mode")
            m_input = input("Select Mode (Default 1): ")
            mode = 'qrcode' if m_input == '2' else 'text'

            tests.run_fft_robustness_test(input_dir=input_dir, mode=mode)

        elif choice.lower() == 'q':
            print("Exiting.")
            break

        else:
            print("Invalid input.")


if __name__ == "__main__":
    main()