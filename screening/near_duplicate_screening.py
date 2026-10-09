import os
import cv2
import csv
import hashlib
from itertools import combinations


TRAIN_DIR = r"..\train\images"
VAL_DIR = r"..\val\images"
TEST_DIR = r"..\test\images"

OUTPUT_CSV = "near_duplicate_pairs.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder, split_name):
    images = []

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()

        if ext not in IMAGE_EXTS:
            continue

        images.append({
            "split": split_name,
            "filename": filename,
            "path": os.path.join(folder, filename)
        })

    return images


def file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def phash(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")

    image = cv2.resize(
        image,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype("float32")
    dct = cv2.dct(image)
    low_freq = dct[:8, :8]
    values = low_freq.flatten()[1:]

    median = float(sorted(values)[len(values) // 2])

    bits = values > median

    hash_value = 0

    for bit in bits:
        hash_value <<= 1

        if bit:
            hash_value |= 1

    return hash_value



def hamming_distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()


train_images = get_images(TRAIN_DIR, "train")
val_images = get_images(VAL_DIR, "val")
test_images = get_images(TEST_DIR, "test")

all_images = train_images + val_images + test_images

print("=" * 60)
print(f"Train: {len(train_images)}")
print(f"Val:   {len(val_images)}")
print(f"Test:  {len(test_images)}")
print(f"Total: {len(all_images)}")
print()



for i, item in enumerate(all_images):

    item["sha256"] = file_hash(item["path"])
    item["phash"] = phash(item["path"])

    if (i + 1) % 200 == 0:
        print(
            f"已处理 {i + 1}/{len(all_images)}"
        )


exact_duplicates = []

for a, b in combinations(all_images, 2):

    if a["split"] == b["split"]:
        continue

    if a["sha256"] == b["sha256"]:

        exact_duplicates.append((a, b))

        print(
        
            f"{a['split']}/{a['filename']}  "
            f"<-> "
            f"{b['split']}/{b['filename']}"
        )

print()
print(
    f"{len(exact_duplicates)}"
)
print()


PHASH_THRESHOLD = 10

print("=" * 60)
print(
    f"(pHash distance <= {PHASH_THRESHOLD})"
)
print("=" * 60)

near_duplicates = []

split_pairs = [
    (train_images, test_images),
    (val_images, test_images)
]

for set_a, set_b in split_pairs:

    for a in set_a:

        for b in set_b:

            distance = hamming_distance(
                a["phash"],
                b["phash"]
            )

            if distance <= PHASH_THRESHOLD:

                near_duplicates.append({
                    "split_a": a["split"],
                    "image_a": a["filename"],
                    "path_a": a["path"],

                    "split_b": b["split"],
                    "image_b": b["filename"],
                    "path_b": b["path"],

                    "phash_distance": distance,

                    "exact_duplicate":
                        a["sha256"] == b["sha256"]
                })

near_duplicates.sort(
    key=lambda x: x["phash_distance"]
)


print(
    f"{len(near_duplicates)}"
)

print()

for item in near_duplicates[:100]:

    print(
        f"d={item['phash_distance']:2d} | "
        f"{item['split_a']}/{item['image_a']} "
        f"<-> "
        f"{item['split_b']}/{item['image_b']}"
    )


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split_a",
            "image_a",
            "path_a",

            "split_b",
            "image_b",
            "path_b",

            "phash_distance",
            "exact_duplicate"
        ]
    )

    writer.writeheader()

    for item in near_duplicates:
        writer.writerow(item)

