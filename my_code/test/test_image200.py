import json
import os
import sys

import faiss
import numpy as np
import torch

# Use the TinyCLIP-bundled open_clip package. Install with `pip install -e .`
# or run this script with the project `src/` directory on PYTHONPATH.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import open_clip
from PIL import Image
from tqdm import tqdm


# Default TinyCLIP checkpoint bundled in `model_hub/`.
# The `pretrained` argument can be either:
#   - a tag string such as "LAION400M" / "LAIONYFCC400M" / "YFCC15M" to
#     auto-download the weights, or
#   - a local path to a `.pt` checkpoint file.
DEFAULT_TINYCLIP_ARCH = "TinyCLIP-ViT-40M-32-Text-19M"
# DEFAULT_TINYCLIP_PRETRAINED = "C:/Users/ASUS/Desktop/大模型代码/TinyCLIP/model_hub/TinyCLIP-ViT-40M-32-Text-19M-LAION400M.pt"
DEFAULT_TINYCLIP_PRETRAINED = "LAION400M"  # 从注册表自动下载权重


def load_tinyclip_model(arch=DEFAULT_TINYCLIP_ARCH, pretrained=DEFAULT_TINYCLIP_PRETRAINED, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # create_model_and_transforms returns (model, preprocess_train, preprocess_val).
    # The original test only needs the val preprocessing pipeline.
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess


def get_tinyclip_tokenizer(arch=DEFAULT_TINYCLIP_ARCH):
    return open_clip.get_tokenizer(arch)


def test_on_imagenet200(model, preprocess, batch_size=64):
    """Zero-shot classification on ImageNet-200 validation set."""
    imagenet_root = "C:/Users/ASUS/Desktop/大模型代码/TinyCLIP/my_code/data/ImageNet-200"
    label_map_path = os.path.join(imagenet_root, "map", "imagenet200_label_map_sorted.json")
    val_anno_path = os.path.join(imagenet_root, "val", "val_annotations.txt")
    val_img_dir = os.path.join(imagenet_root, "val")

    with open(label_map_path, "r") as fp:
        label_data = json.loads(fp.read())

    class_names = [item["class_name"] for item in label_data["items"]]
    wnid_to_label = {item["wnid"]: item["label"] for item in label_data["items"]}

    device = next(model.parameters()).device

    # Step 1: Compute text features for all 200 classes
    print("Computing text features for 200 classes...")
    class_texts = []
    for name in class_names:
        name = name.split(",")[0].strip()
        class_texts.append(f"a photo of a {name}.")

    text_input = get_tinyclip_tokenizer()(class_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features_np = text_features.detach().cpu().numpy().astype('float32')
    print(f"Text features shape: {text_features_np.shape}")

    # Step 2: Load validation annotations
    # Images are in val/{wnid}/images/{img_name}, e.g. val/n03444034/images/val_0.JPEG
    val_annotations = []
    with open(val_anno_path, "r") as fp:
        for line in fp:
            parts = line.strip().split()
            img_name = parts[0]      # e.g. val_0.JPEG
            wnid = parts[1]         # e.g. n03444034
            label = wnid_to_label.get(wnid)
            if label is not None:
                img_path = os.path.join(val_img_dir, wnid, "images", img_name)
                val_annotations.append({"img_path": img_path, "label": label})

    print(f"Loaded {len(val_annotations)} validation samples")

    # Step 3: Batch process images
    top1_correct = 0
    top5_correct = 0
    total = 0

    for i in tqdm(range(0, len(val_annotations), batch_size), desc="Extracting image features"):
        batch_anno = val_annotations[i:i + batch_size]
        batch_images = []
        batch_labels = []
        for anno in batch_anno:
            try:
                img = Image.open(anno["img_path"]).convert("RGB")
                img_input = preprocess(img)
                batch_images.append(img_input)
                batch_labels.append(anno["label"])
            except Exception as e:
                print(f"Error loading {anno['img_path']}: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            img_features = model.encode_image(batch_tensor)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        img_features_np = img_features.detach().cpu().numpy().astype('float32')

        # Compute similarities with all text features
        similarities = img_features_np @ text_features_np.T

        for j, label in enumerate(batch_labels):
            sim = similarities[j]
            top5_indices = np.argsort(sim)[-5:][::-1]
            pred_top1 = top5_indices[0]
            if pred_top1 == label:
                top1_correct += 1
            if label in top5_indices:
                top5_correct += 1
            total += 1

    if total == 0:
        print("No images were successfully loaded!")
        return 0.0, 0.0

    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100
    print(f"\n=== Zero-shot Classification Results ===")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{total})")
    print(f"Top-5 Accuracy: {top5_acc:.2f}% ({top5_correct}/{total})")

    return top1_acc, top5_acc


def test_on_imagenet200_retrieval(model, preprocess, batch_size=64):
    """Text-to-image retrieval on ImageNet-200 validation set."""
    imagenet_root = "C:/Users/ASUS/Desktop/大模型代码/TinyCLIP/my_code/data/ImageNet-200"
    label_map_path = os.path.join(imagenet_root, "map", "imagenet200_label_map_sorted.json")
    val_anno_path = os.path.join(imagenet_root, "val", "val_annotations.txt")
    val_img_dir = os.path.join(imagenet_root, "val")

    with open(label_map_path, "r") as fp:
        label_data = json.loads(fp.read())

    class_names = [item["class_name"] for item in label_data["items"]]
    wnid_to_label = {item["wnid"]: item["label"] for item in label_data["items"]}

    device = next(model.parameters()).device

    # Step 1: Compute text features for all 200 classes
    print("Computing text features for 200 classes...")
    class_texts = []
    for name in class_names:
        name = name.split(",")[0].strip()
        class_texts.append(f"a photo of a {name}.")

    text_input = get_tinyclip_tokenizer()(class_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features_np = text_features.detach().cpu().numpy().astype('float32')

    # Step 2: Load validation annotations
    # Images are in val/{wnid}/images/{img_name}, e.g. val/n03444034/images/val_0.JPEG
    val_annotations = []
    with open(val_anno_path, "r") as fp:
        for line in fp:
            parts = line.strip().split()
            img_name = parts[0]      # e.g. val_0.JPEG
            wnid = parts[1]         # e.g. n03444034
            label = wnid_to_label.get(wnid)
            if label is not None:
                img_path = os.path.join(val_img_dir, wnid, "images", img_name)
                val_annotations.append({"img_path": img_path, "label": label})

    # Step 3: Batch process all images to build image feature index
    print("Extracting image features for retrieval...")
    image_features_list = []
    for i in tqdm(range(0, len(val_annotations), batch_size), desc="Extracting image features"):
        batch_anno = val_annotations[i:i + batch_size]
        batch_images = []
        for anno in batch_anno:
            try:
                img = Image.open(anno["img_path"]).convert("RGB")
                img_input = preprocess(img)
                batch_images.append(img_input)
            except Exception as e:
                print(f"Error loading {anno['img_path']}: {e}")
                continue

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            img_features = model.encode_image(batch_tensor)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        image_features_list.append(img_features.detach().cpu().numpy().astype('float32'))

    image_features_np = np.concatenate(image_features_list, axis=0)
    print(f"Image features shape: {image_features_np.shape}")

    # Step 4: Build FAISS index on image features
    d = image_features_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(image_features_np)

    # Step 5: Text-to-image retrieval: each class query retrieves top-k images
    num_queries = len(class_texts)
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    for query_idx in range(num_queries):
        query_feature = text_features_np[query_idx:query_idx + 1]
        _, indices = index.search(query_feature, k=10)
        retrieved_indices = indices[0].tolist()

        gt_images_per_class = [i for i, anno in enumerate(val_annotations) if anno["label"] == query_idx]
        if not gt_images_per_class:
            continue

        if retrieved_indices[0] in gt_images_per_class:
            top1_correct += 1
        if any(idx in gt_images_per_class for idx in retrieved_indices[:5]):
            top5_correct += 1
        if any(idx in gt_images_per_class for idx in retrieved_indices[:10]):
            top10_correct += 1

    total_valid_queries = sum(1 for i in range(num_queries) if any(a["label"] == i for a in val_annotations))
    if total_valid_queries == 0:
        print("No valid queries found!")
        return 0.0, 0.0, 0.0

    top1_recall = top1_correct / total_valid_queries * 100
    top5_recall = top5_correct / total_valid_queries * 100
    top10_recall = top10_correct / total_valid_queries * 100

    print(f"\n=== Text-to-Image Retrieval Results ===")
    print(f"Total valid class queries: {total_valid_queries}")
    print(f"Top-1 Recall: {top1_recall:.2f}% ({top1_correct}/{total_valid_queries})")
    print(f"Top-5 Recall: {top5_recall:.2f}% ({top5_correct}/{total_valid_queries})")
    print(f"Top-10 Recall: {top10_recall:.2f}% ({top10_correct}/{total_valid_queries})")

    return top1_recall, top5_recall, top10_recall


if __name__ == '__main__':
    # === TinyCLIP model ===
    # `arch` selects the model architecture (must match a config in
    # src/open_clip/model_configs/). `pretrained` can be either a tag string
    # (e.g. "LAION400M") to download weights, or a local path to a .pt file.
    # To switch models, just change the two arguments below. Other TinyCLIP
    # checkpoints are listed in README.md (Model Zoo).
    arch = DEFAULT_TINYCLIP_ARCH
    pretrained = DEFAULT_TINYCLIP_PRETRAINED

    tinyclip_model, preprocess = load_tinyclip_model(arch=arch, pretrained=pretrained)

    print(f"Testing TinyCLIP ({arch}) on ImageNet-200")

    test_on_imagenet200(tinyclip_model, preprocess)
    test_on_imagenet200_retrieval(tinyclip_model, preprocess)
