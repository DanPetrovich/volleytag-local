import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18


SEED = 51
TRAIN_SOURCE, VALIDATION_SOURCE, HELD_SOURCE = 10, 2, 3
MEAN = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
STD = torch.tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)


class Clips(Dataset):
    def __init__(self, clips, labels, indices, augment=False):
        self.clips, self.labels = clips, labels
        self.indices, self.augment = np.asarray(indices), augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = torch.from_numpy(self.clips[self.indices[index]].astype(np.float32) / 255)
        if self.augment:
            if random.random() < .5:
                item = item.flip(-1)
            item = (item * random.uniform(.8, 1.2)).clamp(0, 1)
            if random.random() < .5:
                item = (item + torch.randn_like(item) * .015).clamp(0, 1)
        item = (item - MEAN) / STD
        return item, int(self.labels[self.indices[index]])


def metrics(model, loader, device):
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for clips, labels in loader:
            output = model(clips.to(device)).argmax(1).cpu()
            truth.extend(labels.tolist())
            predicted.extend(output.tolist())
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    specificity = tn / max(tn + fp, 1)
    return {"samples": len(truth), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = next(Path("/kaggle/input").rglob("referee-clips-v2s.npz"))
    data = np.load(data_path)
    clips, sources = data["clips"], data["sources"].astype(int)
    labels = (data["labels"].astype(str) != "no_signal").astype(np.int64)
    train_idx = np.flatnonzero(sources == TRAIN_SOURCE)
    validation_idx = np.flatnonzero(sources == VALIDATION_SOURCE)
    held_idx = np.flatnonzero(sources == HELD_SOURCE)
    train_loader = DataLoader(Clips(clips, labels, train_idx, True), batch_size=8,
                              shuffle=True, num_workers=2, pin_memory=True)
    validation_loader = DataLoader(Clips(clips, labels, validation_idx), batch_size=8,
                                   num_workers=2, pin_memory=True)
    held_loader = DataLoader(Clips(clips, labels, held_idx), batch_size=8,
                             num_workers=2, pin_memory=True)
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    model.to(device)
    counts = np.bincount(labels[train_idx], minlength=2)
    weights = torch.tensor(len(train_idx) / np.maximum(counts, 1),
                           dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)
    history, best_key, best_state, best_epoch = [], None, None, None
    for epoch in range(1, 9):
        model.train(); losses = []
        for batch, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch.to(device)), batch_labels.to(device))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        validation = metrics(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation": validation}
        history.append(row); print(json.dumps(row), flush=True)
        key = (min(validation["recall"], validation["specificity"]),
               validation["balanced_accuracy"], validation["precision"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    held = metrics(model, held_loader, device)
    report = {
        "schema_version": 1, "kind": "volleytag.referee-temporal-finetune-sanity",
        "frozen_tests_used": False, "device": str(device),
        "protocol": {"train_source": TRAIN_SOURCE, "validation_source": VALIDATION_SOURCE,
                     "held_source": HELD_SOURCE, "held_used_once": True,
                     "epochs": 8, "trainable": "r3d18.layer4+fc"},
        "counts": {"train": len(train_idx), "validation": len(validation_idx),
                   "held": len(held_idx)},
        "best_epoch": best_epoch, "history": history, "held": held,
        "acceptance": {"required_recall": .9, "required_specificity": .7,
                       "passed": held["recall"] >= .9 and held["specificity"] >= .7},
    }
    torch.save({"state_dict": best_state, "report": report},
               "/kaggle/working/referee-temporal-finetune-v1.pt")
    Path("/kaggle/working/referee-temporal-finetune-report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
