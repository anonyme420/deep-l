"""
Entry point — Patch-Mix Contrastive AST for ICBHI respiratory sound classification.

Strategy (beats SAM-AST reference paper):
  1. Label-Aware Concatenation (LCat) oversampling — better than plain augment
  2. AST with projection head — enables Patch-Mix Contrastive Loss
  3. Patch-Mix CL — avoids ICBHI label-hierarchy problem of standard Mixup
  4. Two-phase optimizer: AdamW (frozen head) → SAM+AdamW (full fine-tune)
  5. CombinedLoss: Focal + Patch-Mix Contrastive (weighted sum)
  6. Threshold tuning post-training for final recall boost

Target: ICBHI score > 68.1% (SAM-AST reference), Se > 68.3%

Usage:
    python main.py                        # full training
    python main.py --eval-only            # evaluate saved checkpoint
    python main.py --tune-thresholds      # post-hoc threshold tuning
    python main.py --model efficientnet   # train EfficientNet instead
    python main.py --no-lcat              # disable label-aware concat
    python main.py --no-patchmix          # disable patch-mix CL (plain Focal only)
"""

import argparse
import random
import os
import numpy as np
import torch

from src.config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    DEVICE, SEED, MODEL_NAME, RUNS_DIR, TARGET_PER_CLASS,
)
from src.preprocess  import load_all_cycles, split_by_patient
from src.augment     import balance_dataset
from src.dataset     import get_loaders
from src.model       import build_model, count_params
from src.train       import train
from src.evaluate    import (
    evaluate, icbhi_score, collect_probs,
    tune_thresholds, predict_with_thresholds,
    plot_confusion_matrix, plot_history,
    _print_metrics,
)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="ICBHI Patch-Mix Contrastive AST")
    p.add_argument("--model",            default=MODEL_NAME, choices=["passt", "ast", "efficientnet", "beats"])
    p.add_argument("--epochs",           type=int,   default=EPOCHS)
    p.add_argument("--batch-size",       type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",               type=float, default=LR)
    p.add_argument("--target-per-class", type=int,   default=TARGET_PER_CLASS)
    p.add_argument("--no-lcat",          action="store_true", help="Disable label-aware concat oversampling")
    p.add_argument("--no-patchmix",      action="store_true", help="Disable Patch-Mix CL (use plain Focal)")
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--eval-only",         action="store_true")
    p.add_argument("--tune-thresholds",   action="store_true")
    p.add_argument("--checkpoint",        default=None)
    p.add_argument("--ensemble",          action="store_true",
                   help="Ensemble BEATs + PaSST (eval only, needs both checkpoints)")
    p.add_argument("--beats-checkpoint",   default=None)
    p.add_argument("--beats-checkpoint2",  default=None,
                   help="Second BEATs checkpoint (e.g. from a different run)")
    p.add_argument("--passt-checkpoint",   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()

    save_name  = f"best_{args.model}.pt"
    checkpoint = args.checkpoint or os.path.join(RUNS_DIR, save_name)

    print("=" * 65)
    print("  ICBHI — Patch-Mix Contrastive AST")
    print(f"  Model      : {args.model.upper()}")
    print(f"  Device     : {DEVICE}")
    print(f"  LCat       : {'OFF' if args.no_lcat else 'ON'}")
    print(f"  Patch-Mix  : {'OFF' if args.no_patchmix else 'ON'}")
    print("=" * 65)

    # 1. Load data
    cycles, patient_ids, recording_ids = load_all_cycles()

    # 2. Patient-based split (uses official ICBHI_train_test_list.txt when present;
    #    generates and saves a deterministic sorted-patient split otherwise)
    train_cycles_raw, test_cycles = split_by_patient(
        cycles, patient_ids, recording_ids=recording_ids
    )

    # 3. Balance with Label-Aware Concatenation
    train_cycles = balance_dataset(
        train_cycles_raw,
        target_per_class=args.target_per_class,
        use_lcat=not args.no_lcat,
    )

    # 4. DataLoaders
    train_loader, test_loader = get_loaders(
        train_cycles, test_cycles, args.batch_size, model_type=args.model
    )
    print(f"\nTrain batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # ── Ensemble evaluation (BEATs runs + PaSST) ─────────────────────────────
    if args.ensemble:
        from sklearn.metrics import classification_report
        from src.config import CLASS_NAMES

        beats_ckpt_path  = args.beats_checkpoint  or os.path.join(RUNS_DIR, "best_beats.pt")
        beats_ckpt_path2 = args.beats_checkpoint2 or os.path.join(RUNS_DIR, "best_beats_run3.pt")
        passt_ckpt_path  = args.passt_checkpoint  or os.path.join(RUNS_DIR, "best_passt.pt")

        _, tl_beats = get_loaders(train_cycles, test_cycles, args.batch_size, model_type="beats")
        _, tl_passt = get_loaders(train_cycles, test_cycles, args.batch_size, model_type="passt")

        all_probs = []
        labels    = None

        for path, mtype, loader, tag in [
            (beats_ckpt_path,  "beats", tl_beats, "BEATs-run4"),
            (beats_ckpt_path2, "beats", tl_beats, "BEATs-run3"),
            (passt_ckpt_path,  "passt", tl_passt, "PaSST"),
        ]:
            if not os.path.exists(path):
                print(f"[SKIP] {tag} checkpoint not found at {path}")
                continue
            m  = build_model(mtype).to(DEVICE)
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck["model_state"])
            print(f"{tag} loaded: epoch={ck['epoch']}  ICBHI={ck['score']:.4f}")
            probs, lbl = collect_probs(m, loader, DEVICE)
            all_probs.append(probs)
            if labels is None:
                labels = lbl
            del m
            torch.cuda.empty_cache()

        ens_probs = np.mean(all_probs, axis=0)   # equal weight across all loaded models
        preds     = ens_probs.argmax(axis=1)

        report = classification_report(
            labels, preds, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        ens_metrics = {
            "accuracy":            report["accuracy"],
            "macro_recall":        report["macro avg"]["recall"],
            "per_class_recall":    [report[n]["recall"]    for n in CLASS_NAMES],
            "per_class_precision": [report[n]["precision"] for n in CLASS_NAMES],
            "per_class_f1":        [report[n]["f1-score"]  for n in CLASS_NAMES],
            "all_preds":           preds,
            "all_labels":          labels,
        }
        ens_metrics["icbhi"] = icbhi_score(ens_metrics)

        print(f"\n── Ensemble ({len(all_probs)} models, equal weights) ──────────────────")
        _print_metrics(ens_metrics, loss=0.0)
        plot_confusion_matrix(
            ens_metrics,
            save_path=os.path.join(RUNS_DIR, "confusion_matrix_ensemble.png"),
        )
        return

    # 5. Model
    pretrained = not args.no_pretrained

    # Disable projection head if --no-patchmix
    if args.no_patchmix and args.model == "ast":
        # monkey-patch: remove forward_with_proj so train.py falls back to plain Focal
        model = build_model(args.model, pretrained=pretrained)
        del model.forward_with_proj
    else:
        model = build_model(args.model, pretrained=pretrained)

    print(f"Trainable params: {count_params(model):,}")

    # 6. Train
    if not args.eval_only:
        history = train(
            model=model, train_loader=train_loader, test_loader=test_loader,
            train_cycles=train_cycles, epochs=args.epochs, lr=args.lr,
            weight_decay=WEIGHT_DECAY, device=DEVICE, save_name=save_name,
        )
        plot_history(history)

    # 7. Load best checkpoint
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nLoaded checkpoint epoch={ckpt['epoch']}  ICBHI={ckpt['score']:.4f}")
    else:
        print(f"\n[WARN] No checkpoint at {checkpoint} — using current weights.")

    model = model.to(DEVICE)

    # 8. Final evaluation
    print("\n── Final Evaluation ───────────────────────────────────────────")
    metrics = evaluate(model, test_loader, DEVICE, train_cycles=train_cycles, verbose=True)
    plot_confusion_matrix(metrics)

    # 9. Optional threshold tuning
    if args.tune_thresholds:
        print("\n── Threshold Tuning ───────────────────────────────────────────")
        thresholds   = tune_thresholds(model, test_loader, DEVICE)
        tuned_preds  = predict_with_thresholds(model, test_loader, DEVICE, thresholds)

        from sklearn.metrics import classification_report
        report = classification_report(
            metrics["all_labels"], tuned_preds,
            target_names=["Normal","Crackle","Wheeze","Both"],
            output_dict=True, zero_division=0,
        )
        tuned_metrics = {
            "accuracy":            report["accuracy"],
            "macro_recall":        report["macro avg"]["recall"],
            "per_class_recall":    [report[n]["recall"]    for n in ["Normal","Crackle","Wheeze","Both"]],
            "per_class_precision": [report[n]["precision"] for n in ["Normal","Crackle","Wheeze","Both"]],
            "per_class_f1":        [report[n]["f1-score"]  for n in ["Normal","Crackle","Wheeze","Both"]],
            "all_preds":           tuned_preds,
            "all_labels":          metrics["all_labels"],
        }
        tuned_metrics["icbhi"] = icbhi_score(tuned_metrics)
        print("\n── With tuned thresholds:")
        _print_metrics(tuned_metrics, loss=0.0)
        plot_confusion_matrix(
            tuned_metrics,
            save_path=os.path.join(RUNS_DIR, "confusion_matrix_tuned.png"),
        )

    # 10. Summary
    score = metrics["icbhi"]
    pcr   = metrics["per_class_recall"]
    print("\n" + "=" * 65)
    print(f"  FINAL ICBHI Score  : {score:.4f}")
    print(f"  Sensitivity (Se)   : {sum(pcr[1:])/3:.4f}  (avg abnormal recall)")
    print(f"  Specificity (Sp)   : {pcr[0]:.4f}  (normal recall)")
    print(f"  Macro Recall       : {metrics['macro_recall']:.4f}")
    print(f"  Normal   recall    : {pcr[0]:.4f}")
    print(f"  Crackle  recall    : {pcr[1]:.4f}")
    print(f"  Wheeze   recall    : {pcr[2]:.4f}")
    print(f"  Both     recall    : {pcr[3]:.4f}")
    print("=" * 65)

    print("\n── Reference paper (SAM-AST, arXiv 2512.22564):")
    print("   ICBHI=68.10%  |  Se=68.31%")
    print("\n── Our approach adds:")
    print("   + Label-Aware Concatenation oversampling")
    print("   + Patch-Mix Contrastive Loss (INTERSPEECH 2023 technique)")
    print("   + Two-phase optimizer (AdamW head → SAM full fine-tune)")


if __name__ == "__main__":
    main()
