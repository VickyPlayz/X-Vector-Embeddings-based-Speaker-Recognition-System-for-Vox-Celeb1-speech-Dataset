import os

DATA_ROOT = r'G:\3 x vector with 10 percent dataset\Dataset\VoxCeleb\vox1_dev_wav'

print(f"Checking {DATA_ROOT}...")
if not os.path.exists(DATA_ROOT):
    print("ROOT DOES NOT EXIST!")
    exit(1)

try:
    items = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f"Found {len(items)} speaker directories.")
    if len(items) > 0:
        print(f"First 5: {items[:5]}")
        print(f"Last 5: {items[-5:]}")

except Exception as e:
    print(f"Error: {e}")
