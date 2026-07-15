from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .artifacts import build_test_dataset, build_train_dataset, load_beam_head, save_checkpoint
from .config import (
    DATA_ROOT_DEFAULT,
    DEEPSENSE_ROOT_DEFAULT,
    FORMAL_BACKBONE,
    FORMAL_CODEBOOK_FRAME,
    FORMAL_DEEPSENSE_LABEL_SOURCE,
    FORMAL_LAMBDA_LABEL_MODE,
    FORMAL_TEST_DATASET,
    FORMAL_TRAIN_SCENE,
)
from .data import DeepSenseRGB60BeamDataset
from .geometry import PhotoULACodebook
from .models import RGBBeamNet, assert_model_structure, model_file_stem
from .sampling import (
    few_shot_finetune_deepsense,
    load_few_shot_manifest,
    sample_few_shot_indices,
    save_few_shot_manifest,
)
from .training import (
    class_weights_from_labels,
    evaluate,
    label_summary,
    make_loader_kwargs,
    split_indices,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default=DATA_ROOT_DEFAULT,
        required=DATA_ROOT_DEFAULT is None,
        help="Prepared LAMBDA dataset root, or set LAMBDA_DATA_ROOT.",
    )
    parser.add_argument(
        "--deepsense-root",
        "--deepsense_root",
        dest="deepsense_root",
        default=DEEPSENSE_ROOT_DEFAULT,
        required=DEEPSENSE_ROOT_DEFAULT is None,
        help="DeepSense Scenario 23 root, or set DEEPSENSE_ROOT.",
    )
    parser.add_argument("--deepsense_csv", default="scenario23.csv")
    parser.add_argument("--deepsense_label_source", choices=[FORMAL_DEEPSENSE_LABEL_SOURCE], default=FORMAL_DEEPSENSE_LABEL_SOURCE)
    parser.add_argument("--train_scenes", nargs="+", choices=[FORMAL_TRAIN_SCENE], default=[FORMAL_TRAIN_SCENE])
    parser.add_argument("--test_dataset", choices=[FORMAL_TEST_DATASET], default=FORMAL_TEST_DATASET)
    parser.add_argument("--n_ant", type=int, default=16)
    parser.add_argument("--n_beams", type=int, default=64)
    parser.add_argument("--codebook_fov", type=float, default=90.0)
    parser.add_argument(
        "--lambda_label_mode",
        choices=[FORMAL_LAMBDA_LABEL_MODE],
        default=FORMAL_LAMBDA_LABEL_MODE,
        help="Map the strongest 60 GHz path AoA to the 64-beam label space.",
    )
    parser.add_argument(
        "--lambda_codebook_frame",
        choices=[FORMAL_CODEBOOK_FRAME],
        default=FORMAL_CODEBOOK_FRAME,
        help="Use the formal sky-up ULA codebook while preserving image-horizontal beam ordering.",
    )
    parser.add_argument("--stride", type=int, default=3, help="LAMBDA train frame stride.")
    parser.add_argument("--test_stride", type=int, default=1, help="DeepSense test frame stride.")
    parser.add_argument("--limit_train_per_scene", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--center_codebook_crop",
        action="store_true",
        help="Deterministically crop the central codebook FoV before resizing.",
    )
    parser.add_argument(
        "--rgb_hfov",
        type=float,
        default=110.0,
        help="Assumed RGB horizontal FoV used by --center_codebook_crop.",
    )
    parser.add_argument("--backbone", choices=[FORMAL_BACKBONE], default=FORMAL_BACKBONE)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "paper_step"], default="cosine")
    parser.add_argument("--lr_milestones", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--ft_optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--ft_scheduler", choices=["cosine", "paper_step"], default="cosine")
    parser.add_argument("--ft_lr_milestones", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--ft_lr_gamma", type=float, default=0.1)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--restore_best_val",
        action="store_true",
        help="After LAMBDA training, restore the epoch with the best validation metric before saving/evaluating.",
    )
    parser.add_argument(
        "--best_metric",
        default="top1",
        choices=["loss", "top1", "top3", "top5", "within_1_beam", "within_3_beams", "mae_beams"],
    )
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="runs/cache_rgb60")
    parser.add_argument("--save_dir", default="runs/beam_rgb60")
    parser.add_argument("--do_zero_shot", action="store_true", default=True)
    parser.add_argument("--no_zero_shot", dest="do_zero_shot", action="store_false")
    parser.add_argument("--do_few_shot", action="store_true")
    parser.add_argument("--k_shots", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument(
        "--few_shot_sampling",
        choices=["stratified", "coverage_distribution"],
        default="stratified",
        help=(
            "DeepSense few-shot sampling. coverage_distribution covers all "
            "present labels when K is small enough and otherwise matches the "
            "full label distribution with rounded per-label quotas."
        ),
    )
    parser.add_argument("--few_shot_manifest", default=None)
    parser.add_argument("--write_few_shot_manifest", default=None)
    parser.add_argument("--ft_epochs", type=int, default=8)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--few_shot_save_ckpts", action="store_true")
    parser.add_argument(
        "--few_shot_reset_head",
        dest="few_shot_reset_head",
        action="store_true",
        default=True,
        help="Reinitialize the 64-way beam head before each DeepSense K-shot fine-tune.",
    )
    parser.add_argument(
        "--no_few_shot_reset_head",
        dest="few_shot_reset_head",
        action="store_false",
        help="Keep the LAMBDA-trained beam head for DeepSense K-shot fine-tuning.",
    )
    parser.add_argument("--no_few_shot_auto_schedule", dest="few_shot_auto_schedule", action="store_false")
    parser.set_defaults(few_shot_auto_schedule=True)
    parser.add_argument("--load_ckpt", default=None)
    parser.add_argument("--load_head_ckpt", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.load_ckpt and args.load_head_ckpt:
        raise ValueError("Use only one of --load_ckpt or --load_head_ckpt.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(
        f"Codebook: 1x{args.n_ant} ULA, {args.n_beams} beams over "
        f"{args.codebook_fov:.1f} deg, beam0=photo-left, beam{args.n_beams-1}=photo-right"
    )

    codebook = PhotoULACodebook(
        n_ant=args.n_ant,
        n_beams=args.n_beams,
        fov_deg=args.codebook_fov,
    )
    train_full, train_labels, train_scene_info = build_train_dataset(args, codebook, augment=True)
    val_full, _, _ = build_train_dataset(args, codebook, augment=False, log=False)
    test_ds, test_labels, test_info = build_test_dataset(args)

    print("Combined train label summary:", label_summary(train_labels))
    train_idx, val_idx = split_indices(len(train_full), args.val_ratio, args.seed)
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx) if val_idx else None
    loader_kwargs = make_loader_kwargs(args.num_workers, device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if val_ds is not None
        else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = RGBBeamNet(
        n_classes=args.n_beams,
        backbone=args.backbone,
    )
    assert_model_structure(model, args.n_beams)
    print(
        "Paper RGB model structure: torchvision ResNet-50 backbone + "
        f"Linear(2048, {args.n_beams}) task head, end-to-end fine-tuning."
    )
    class_weights = class_weights_from_labels(train_labels, args.n_beams) if args.weighted_loss else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = model_file_stem(args.backbone)
    ckpt_path = save_dir / f"{artifact_stem}.pt"

    summary: Dict[str, object] = {
        "task": "RGB photo -> 60GHz beam",
        "model": {
            "backbone": args.backbone,
            "pretrained": True,
            "head": "Linear(2048, n_beams)",
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "lr_milestones": args.lr_milestones,
            "lr_gamma": args.lr_gamma,
            "ft_optimizer": args.ft_optimizer,
            "ft_scheduler": args.ft_scheduler,
            "ft_lr_milestones": args.ft_lr_milestones,
            "ft_lr_gamma": args.ft_lr_gamma,
        },
        "beam_order": "beam0 is photo-left, beam63 is photo-right",
        "lambda_label_mode": args.lambda_label_mode,
        "lambda_codebook_frame": args.lambda_codebook_frame,
        "initialization": (
            "full_checkpoint" if args.load_ckpt else
            "imagenet_backbone_lambda_head" if args.load_head_ckpt else
            "imagenet_backbone_random_head"
        ),
        "codebook": {
            "array": f"1x{args.n_ant} ULA",
            "n_beams": args.n_beams,
            "fov_deg": args.codebook_fov,
            "beam_width_deg": codebook.beam_width_deg,
            "beam_centers_deg": codebook.centers_deg.tolist(),
            "center_codebook_crop": bool(args.center_codebook_crop),
            "rgb_hfov_deg": args.rgb_hfov,
            "img_size": args.img_size,
        },
        "train_scenes": train_scene_info,
        "train_combined": label_summary(train_labels),
        "test": test_info,
        "n_train_frames": len(train_ds),
        "n_val_frames": len(val_ds) if val_ds is not None else 0,
        "n_test_frames": len(test_ds),
        "few_shot_reset_head": bool(args.few_shot_reset_head),
        "few_shot_sampling": args.few_shot_sampling,
        "few_shot_manifest": args.few_shot_manifest,
    }

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {args.load_ckpt}")
    else:
        if args.load_head_ckpt:
            load_beam_head(model, args.load_head_ckpt, device)
        if args.epochs > 0:
            print(f"Training: {len(train_ds)} train / {len(val_ds) if val_ds is not None else 0} val")
            best_val_metrics = train_model(
                model,
                train_loader,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                weight_decay=args.weight_decay,
                class_weights=class_weights,
                log_prefix="[train] ",
                amp=not args.no_amp,
                log_interval=args.log_interval,
                epoch_log_interval=1,
                restore_best=args.restore_best_val,
                best_metric=args.best_metric,
                scheduler_name=args.scheduler,
                lr_milestones=args.lr_milestones,
                lr_gamma=args.lr_gamma,
                optimizer_name=args.optimizer,
            )
            if best_val_metrics is not None:
                summary["best_val"] = best_val_metrics
                summary["selected_model"] = f"best_val_{args.best_metric}"
        else:
            print("Skipping LAMBDA training because --epochs 0.")
        save_checkpoint(ckpt_path, model, args, summary)
        print(f"Saved checkpoint -> {ckpt_path}")

    if val_loader is not None:
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"[val] top1={val_metrics['top1']:.4f} top3={val_metrics['top3']:.4f} "
            f"top5={val_metrics['top5']:.4f} within1={val_metrics['within_1_beam']:.4f} "
            f"within3={val_metrics['within_3_beams']:.4f} "
            f"mae={val_metrics['mae_beams']:.2f} bias={val_metrics['bias_beams']:.2f} "
            f"loss={val_metrics['loss']:.4f} n={val_metrics['n']}"
        )
        summary["val"] = val_metrics

    if args.do_zero_shot:
        test_metrics = evaluate(model, test_loader, device)
        print(
            f"[zero-shot DeepSense] top1={test_metrics['top1']:.4f} "
            f"top3={test_metrics['top3']:.4f} top5={test_metrics['top5']:.4f} "
            f"within1={test_metrics['within_1_beam']:.4f} "
            f"within3={test_metrics['within_3_beams']:.4f} "
            f"mae={test_metrics['mae_beams']:.2f} bias={test_metrics['bias_beams']:.2f} "
            f"loss={test_metrics['loss']:.4f} n={test_metrics['n']}"
        )
        summary["zero_shot_deepsense"] = test_metrics

    if args.do_few_shot:
        print("Running DeepSense few-shot fine-tuning ...")
        deepsense_ft = DeepSenseRGB60BeamDataset(
            args.deepsense_root,
            csv_name=args.deepsense_csv,
            stride=args.test_stride,
            img_size=args.img_size,
            augment=True,
            center_codebook_crop=args.center_codebook_crop,
            rgb_hfov=args.rgb_hfov,
            codebook_fov=args.codebook_fov,
            label_source=args.deepsense_label_source,
            limit=args.limit_test,
        )
        manifest_indices = load_few_shot_manifest(args.few_shot_manifest) if args.few_shot_manifest else None
        if args.write_few_shot_manifest:
            labels = np.asarray(getattr(test_ds, "labels"), dtype=np.int64)
            generated = {
                int(k): sample_few_shot_indices(labels, int(k), args.seed + int(k), args.few_shot_sampling)
                for k in args.k_shots
                if 0 < int(k) < len(labels)
            }
            save_few_shot_manifest(args.write_few_shot_manifest, labels, generated)
            print(f"Wrote few-shot manifest -> {args.write_few_shot_manifest}")
            manifest_indices = generated
        few_shot = few_shot_finetune_deepsense(
            model,
            deepsense_ft,
            test_ds,
            k_shots=args.k_shots,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
            weight_decay=args.weight_decay,
            amp=not args.no_amp,
            log_interval=args.log_interval,
            auto_schedule=args.few_shot_auto_schedule,
            reset_head=args.few_shot_reset_head,
            sampling_strategy=args.few_shot_sampling,
            manifest_indices=manifest_indices,
            scheduler_name=args.ft_scheduler,
            lr_milestones=args.ft_lr_milestones,
            lr_gamma=args.ft_lr_gamma,
            optimizer_name=args.ft_optimizer,
            checkpoint_stem=artifact_stem,
            save_dir=save_dir if args.few_shot_save_ckpts else None,
        )
        summary["few_shot_deepsense"] = few_shot

    out_json = save_dir / f"{artifact_stem}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {out_json}")
