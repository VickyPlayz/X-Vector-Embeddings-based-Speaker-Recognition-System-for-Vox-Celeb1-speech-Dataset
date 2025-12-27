import os

DATA_ROOT = r'G:\3 x vector with 10 percent dataset\Dataset\VoxCeleb'

print(f"Checking {DATA_ROOT}...")
if not os.path.exists(DATA_ROOT):
    print("ROOT DOES NOT EXIST!")
    exit(1)

print("Listing top level items:")
try:
    items = os.listdir(DATA_ROOT)
    print(f"Found {len(items)} items.")
    print(items[:10])

    wav_count = 0
    dirs_checked = 0
    
    print("\nDeep search for .wav files...")
    for root, dirs, files in os.walk(DATA_ROOT):
        dirs_checked += 1
        for f in files:
            if f.endswith('.wav'):
                wav_count += 1
                if wav_count <= 5:
                    print(f"Found wav: {os.path.join(root, f)}")
        
        if dirs_checked > 20 and wav_count == 0:
            print(f"Checked {dirs_checked} directories, still no wav found. Current dir: {root}")
            print("Files here:", files)
        
        if wav_count > 0 and dirs_checked > 100:
             break
             
    print(f"\nTotal wavs found (scan limited): {wav_count}")

except Exception as e:
    print(f"Error: {e}")
