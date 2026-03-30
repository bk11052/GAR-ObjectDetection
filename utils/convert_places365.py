#!/usr/bin/env python3
"""Convert Places365 VGG-16 weights from Caffe (.caffemodel) to PyTorch (.pt).

Places365 only provides VGG-16 in Caffe format. This script parses the
caffemodel protobuf binary directly (no Caffe installation required) and
maps weights to torchvision VGG-16 key names.

Usage:
    python utils/convert_places365.py [--output PATH]
"""

import os
import struct
import urllib.request

import numpy as np
import torch

CAFFEMODEL_URL = (
    "http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel"
)

# Caffe layer name → torchvision VGG-16 key prefix
LAYER_MAP = {
    "conv1_1": "features.0",
    "conv1_2": "features.2",
    "conv2_1": "features.5",
    "conv2_2": "features.7",
    "conv3_1": "features.10",
    "conv3_2": "features.12",
    "conv3_3": "features.14",
    "conv4_1": "features.17",
    "conv4_2": "features.19",
    "conv4_3": "features.21",
    "conv5_1": "features.24",
    "conv5_2": "features.26",
    "conv5_3": "features.28",
    "fc6": "classifier.0",
    "fc7": "classifier.3",
    "fc8a": "classifier.6",
}


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format parser (no .proto compilation needed)
# ---------------------------------------------------------------------------

def _read_varint(data, pos):
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _parse_fields(data):
    """Parse protobuf binary into {field_number: [(wire_type, value)]}."""
    pos = 0
    fields = {}
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 0:          # varint
            value, pos = _read_varint(data, pos)
        elif wire_type == 1:        # 64-bit fixed
            value = data[pos:pos + 8]
            pos += 8
        elif wire_type == 2:        # length-delimited
            length, pos = _read_varint(data, pos)
            value = data[pos:pos + length]
            pos += length
        elif wire_type == 5:        # 32-bit fixed
            value = data[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unknown wire type {wire_type}")

        fields.setdefault(field_num, []).append((wire_type, value))
    return fields


def _parse_blob(blob_data):
    """Parse a Caffe BlobProto → numpy array."""
    fields = _parse_fields(blob_data)

    # --- shape ---
    shape = None
    # V2: BlobShape at field 7
    if 7 in fields:
        shape_fields = _parse_fields(fields[7][0][1])
        if 1 in shape_fields:
            dims = []
            for wt, val in shape_fields[1]:
                if wt == 2:             # packed repeated int64
                    p = 0
                    while p < len(val):
                        d, p = _read_varint(val, p)
                        dims.append(d)
                elif wt == 0:           # unpacked varint
                    dims.append(val)
            shape = tuple(dims)

    # V1 fallback: num(1) channels(2) height(3) width(4)
    if shape is None:
        v1 = []
        for fid in (1, 2, 3, 4):
            if fid in fields and fields[fid][0][0] == 0:
                v1.append(fields[fid][0][1])
        if v1:
            shape = tuple(d for d in v1 if d > 0)

    # --- float data (field 5) ---
    if 5 not in fields:
        return None

    chunks = []
    for wt, val in fields[5]:
        if wt == 2:                     # packed floats
            chunks.append(np.frombuffer(val, dtype=np.float32))
        elif wt == 5:                   # single 32-bit float
            chunks.append(np.frombuffer(val, dtype=np.float32))
    arr = np.concatenate(chunks).copy()

    if shape is not None:
        total = 1
        for d in shape:
            total *= d
        if arr.size == total:
            arr = arr.reshape(shape)

    return arr


def parse_caffemodel(path):
    """Parse .caffemodel → {layer_name: [weight, bias, ...]}."""
    with open(path, "rb") as f:
        data = f.read()
    net = _parse_fields(data)
    layers = {}

    # V2 format: field 100 = repeated LayerParameter
    #   LayerParameter: name=1, blobs=7
    if 100 in net:
        for _, layer_data in net[100]:
            if not isinstance(layer_data, (bytes, bytearray)):
                continue
            try:
                lf = _parse_fields(layer_data)
            except Exception:
                continue
            if 1 not in lf:
                continue
            name = lf[1][0][1].decode("utf-8", errors="ignore")
            if 7 not in lf:
                continue
            blobs = []
            for _, bd in lf[7]:
                arr = _parse_blob(bd)
                if arr is not None:
                    blobs.append(arr)
            if blobs:
                layers[name] = blobs

    # V1 fallback: field 2 = repeated V1LayerParameter
    #   V1LayerParameter: name=4, blobs=6
    if not layers and 2 in net:
        for _, layer_data in net[2]:
            if not isinstance(layer_data, (bytes, bytearray)):
                continue
            try:
                lf = _parse_fields(layer_data)
            except Exception:
                continue
            if 4 not in lf:
                continue
            name = lf[4][0][1].decode("utf-8", errors="ignore")
            if 6 not in lf:
                continue
            blobs = []
            for _, bd in lf[6]:
                arr = _parse_blob(bd)
                if arr is not None:
                    blobs.append(arr)
            if blobs:
                layers[name] = blobs

    return layers


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_pytorch(caffe_layers):
    """Map Caffe layer weights to torchvision VGG-16 state dict keys."""
    state_dict = {}
    for caffe_name, pytorch_prefix in LAYER_MAP.items():
        if caffe_name not in caffe_layers:
            print(f"  WARNING: {caffe_name} not found in caffemodel")
            continue
        blobs = caffe_layers[caffe_name]
        if len(blobs) >= 1:
            state_dict[f"{pytorch_prefix}.weight"] = torch.from_numpy(blobs[0])
        if len(blobs) >= 2:
            state_dict[f"{pytorch_prefix}.bias"] = torch.from_numpy(blobs[1])
        print(f"  {caffe_name} → {pytorch_prefix}  "
              f"weight={blobs[0].shape}"
              f"{f'  bias={blobs[1].shape}' if len(blobs) > 1 else ''}")
    return state_dict


def download_and_convert(output_path):
    """Download caffemodel, convert, and save as .pt."""
    cache_dir = os.path.dirname(output_path)
    os.makedirs(cache_dir, exist_ok=True)

    caffemodel_path = os.path.join(cache_dir, "vgg16_places365.caffemodel")

    # Download .caffemodel
    if not os.path.exists(caffemodel_path):
        print(f"[Places365] Downloading caffemodel → {caffemodel_path}")
        urllib.request.urlretrieve(CAFFEMODEL_URL, caffemodel_path)
        print(f"[Places365] Download complete ({os.path.getsize(caffemodel_path) / 1e6:.0f} MB)")
    else:
        print(f"[Places365] Using cached caffemodel: {caffemodel_path}")

    # Parse & convert
    print("[Places365] Parsing caffemodel...")
    caffe_layers = parse_caffemodel(caffemodel_path)
    print(f"[Places365] Found {len(caffe_layers)} layers: {list(caffe_layers.keys())}")

    print("[Places365] Converting to PyTorch format...")
    state_dict = convert_to_pytorch(caffe_layers)

    # Save
    torch.save(state_dict, output_path)
    n_params = sum(v.numel() for v in state_dict.values())
    print(f"[Places365] Saved → {output_path} ({n_params:,} parameters)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert Places365 VGG-16 Caffe→PyTorch")
    parser.add_argument("--output", default=os.path.expanduser("~/.cache/places365/vgg16_places365.pt"))
    args = parser.parse_args()
    download_and_convert(args.output)


if __name__ == "__main__":
    main()
