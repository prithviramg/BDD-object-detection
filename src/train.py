import os
import math
import argparse
import torch
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Import your modules
from model import ObjectDetectionModel
from loss import MultiScaleCenterNetLoss
from dataset import BDDDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iteration-Based Training with Validation, TensorBoard Logging, and ReduceLROnPlateau Scheduler"
    )
    parser.add_argument(
        "--data_dir",
        default=r"D:\database\bdd_dataset\bdd100k_images_100k\bdd100k\images\100k\train",
        help="Path to images directory.",
    )
    parser.add_argument(
        "--anno_file",
        default=r"D:\database\bdd_dataset\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json",
        help="Path to annotation JSON file.",
    )
    parser.add_argument(
        "--val_data_dir",
        default=r"D:\database\bdd_dataset\bdd100k_images_100k\bdd100k\images\100k\val",
        help="Path to images directory.",
    )
    parser.add_argument(
        "--val_anno_file",
        default=r"D:\database\bdd_dataset\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json",
        help="Path to annotation JSON file.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[720, 1280],
        help="Input image size: height width",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--max_objs", type=int, default=64, help="Max number of objects per image."
    )
    parser.add_argument(
        "--save_dir",
        default="./checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Factor by which the learning rate will be reduced.",
    )
    # Validation and iteration parameters.
    parser.add_argument(
        "--max_iters",
        type=int,
        default=140000,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=100,
        help="Frequency (in iterations) to run validation.",
    )
    # TensorBoard log directory.
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory for TensorBoard logs."
    )
    args = parser.parse_args()
    return args


def evaluate(model, dataloader, device, loss_fn_tc, loss_fn_tp):
    model.eval()
    total_loss = 0.0
    print("evaluating validation model")
    with torch.no_grad():
        for images, targets in tqdm.tqdm(dataloader):
            images = images.to(device)
            targets_tc = {key: targets["tc"][key].to(device) for key in targets["tc"]}
            targets_tp = {key: targets["tp"][key].to(device) for key in targets["tp"]}

            outputs = model(images)
            outputs_tc = (
                outputs["tc"]["heatmap"],
                outputs["tc"]["size"],
                outputs["tc"]["offset"],
            )
            outputs_tp = (
                outputs["tp"]["heatmap"],
                outputs["tp"]["size"],
                outputs["tp"]["offset"],
            )

            loss_tc = loss_fn_tc([outputs_tc], [targets_tc])
            loss_tp = loss_fn_tp([outputs_tp], [targets_tp])
            loss = loss_tc + loss_tp

            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    model.train()  # Return model to training mode after evaluation.
    return avg_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create TensorBoard writer.
    writer = SummaryWriter(log_dir=args.log_dir)

    # Create dataset and perform train/val split.
    print("loading train annotations. !!!")
    train_dataset = BDDDataset(
        image_dir=args.data_dir,
        annotation_file=args.anno_file,
        input_size=tuple(args.input_size),
        max_objs=args.max_objs,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    print("loading validation annotations. !!!")
    val_dataset = BDDDataset(
        image_dir=args.val_data_dir,
        annotation_file=args.val_anno_file,
        input_size=tuple(args.input_size),
        max_objs=args.max_objs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(val_dataset, replacement=True, num_samples=500),
        collate_fn=collate_fn,
    )

    # Initialize model, loss functions, optimizer, and scheduler.
    model = ObjectDetectionModel(tp_class=8, tc_class=2, bifpn_channels=256)
    model = model.to(device)

    loss_fn_tc = MultiScaleCenterNetLoss(wh_weight=0.1, off_weight=1.0)
    loss_fn_tp = MultiScaleCenterNetLoss(wh_weight=0.1, off_weight=1.0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler that reduces LR when validation loss plateaus.
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(args.save_dir, exist_ok=True)

    # Iteration-based training.
    iteration = 0
    train_loader_iter = iter(train_loader)

    for iteration in range(args.max_iters):
        try:
            images, targets = next(train_loader_iter)
        except StopIteration:
            # Reinitialize iterator if the dataloader is exhausted.
            train_loader_iter = iter(train_loader)
            images, targets = next(train_loader_iter)

        images = images.to(device)
        targets_tc = {key: targets["tc"][key].to(device) for key in targets["tc"]}
        targets_tp = {key: targets["tp"][key].to(device) for key in targets["tp"]}

        optimizer.zero_grad()
        outputs = model(images)
        outputs_tc = (
            outputs["tc"]["heatmap"],
            outputs["tc"]["size"],
            outputs["tc"]["offset"],
        )
        outputs_tp = (
            outputs["tp"]["heatmap"],
            outputs["tp"]["size"],
            outputs["tp"]["offset"],
        )

        loss_tc = loss_fn_tc([outputs_tc], [targets_tc])
        loss_tp = loss_fn_tp([outputs_tp], [targets_tp])
        loss = loss_tc + loss_tp

        loss.backward()
        optimizer.step()

        # Log training loss and current learning rate to TensorBoard.
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", loss.item(), iteration)
        writer.add_scalar("LearningRate", current_lr, iteration)

        # Print training progress every 10 iterations.
        if iteration % 10 == 0:
            print(
                f"Iteration [{iteration}/{args.max_iters}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )

        # Run validation every 'val_freq' iterations.
        if iteration % args.val_freq == 0:
            val_loss = evaluate(model, val_loader, device, loss_fn_tc, loss_fn_tp)
            print(f"*** Iteration [{iteration}] Validation Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/validation", val_loss, iteration)

            # Step the scheduler using the validation loss.
            scheduler.step()

            # Optionally save a checkpoint.
            checkpoint_path = os.path.join(args.save_dir, f"model_iter_{iteration}.pth")
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    # Close the TensorBoard writer after training.
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
