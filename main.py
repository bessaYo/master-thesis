"""
Runs a single slice and prints a detailed report

Usage:
    python main.py --model resnet-cifar --target 0
    python main.py --model resnet18 --target 0 --channel_alpha 0.8
"""

import argparse

import torch

from models import get_model
from core.slicer import Slicer
from utils.data import (
    CIFAR10_CLASSES,
    load_cifar10,
    load_imagenet_val,
)
from utils.report import (
    print_header,
    print_neuron_table,
    print_block_analysis,
    print_slice_summary,
)


def main():
    parser = argparse.ArgumentParser(description="NNSlicer — detailed slicing report")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "resnet-cifar",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        ],
        help="Model name",
    )
    parser.add_argument("--target", type=int, required=True, help="Target class index")
    parser.add_argument("--theta", type=float, default=0.3, help="Contribution threshold")
    parser.add_argument("--channel_alpha", type=float, default=None, help="Channel energy fraction (enables channel filtering)")
    parser.add_argument("--block_beta", type=float, default=None, help="Block energy fraction (enables block filtering)")
    parser.add_argument(
        "--image_index",
        type=int,
        default=0,
        help="Index of the test image in the dataset",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and class names per model
    if args.model == "resnet-cifar":
        dataset = load_cifar10(train=False)
        class_names = CIFAR10_CLASSES
        profile_path = "pretrained/profiles/cifar10_resnet-cifar.pt"
        dataset_name = "cifar10"
    elif args.model in ("resnet18", "resnet34", "resnet50", "resnet101"):
        dataset = load_imagenet_val()
        class_names = None
        profile_path = f"pretrained/profiles/imagenet_{args.model}.pt"
        dataset_name = "imagenet"
    target_idx = args.target

    # Load model and profile
    model = get_model(args.model, pretrained=True).to(device).eval()
    profile = torch.load(profile_path, map_location=device, weights_only=False)

    # Load sample image
    image, label = dataset[args.image_index]
    inp = image.unsqueeze(0).to(device)
    slicer = Slicer(
        model=model,
        input_sample=inp,
        precomputed_profile=profile,
    )

    slicer.profile()
    slicer.forward()

    slicer.backward(
        target_index=target_idx,
        theta=args.theta,
        channel_alpha=args.channel_alpha,
        block_beta=args.block_beta,
    )

    nc = slicer.backward_result["neuron_contributions"]
    t_backward = slicer.backward_result["backward_time"]

    # Print report
    print_header(
        args.model,
        model,
        dataset_name,
        class_names,
        target_idx,
        label,
        args.image_index,
        args.theta,
        args.channel_alpha,
        args.block_beta,
    )
    print_neuron_table(nc, model)

    if slicer.forward_result.get("blocks"):
        print_block_analysis(slicer.forward_result, slicer.backward_result, nc)

    print_slice_summary(slicer.backward_result, model, nc, t_backward)


if __name__ == "__main__":
    main()
