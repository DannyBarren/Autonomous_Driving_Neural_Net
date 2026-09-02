# Architecture

How the repo is laid out and what the training script actually does.

## Files

| Path | What it is |
| --- | --- |
| `Barren_Object_Detection_Neural_Network.py` | The whole pipeline. Label loading, box cleaning, split, dataset, model, train loop, weight save. Single file, guarded by `if __name__ == '__main__':` so Windows `spawn` does not re-execute it. |
| `requirements.txt` | Runtime deps, taken from the imports in the script. |
| `docs/ARCHITECTURE.md` | This file. |
| `README.md` | Overview, reported results, how to train. |
| `.gitignore` | Keeps images, `labels.csv`, `.pth` weights, and `inferences/` out of git. |
| `LICENSE` | MIT, 2025. |

## Flow inside the script

1. **Device pick.** `cuda` if available, else `cpu`. Prints the GPU name.
2. **`IMAGE_DIR`.** A blank assignment in the committed file. The operator sets it locally.
3. **Labels.** `pandas.read_csv('labels.csv', header=None, ...)` with columns `image_id, class, xmin, ymin, xmax, ymax`. `image_id` is cast to string and zero-padded to 8 digits so it matches filenames like `00000156.jpg`.
4. **Global box clean.** Computes `width` and `height`, prints any rows where either is `<= 0`, then drops those rows.
5. **Image pre-filter.** Walks the unique `image_id` values and keeps only the ones where `<IMAGE_DIR>/<image_id>.jpg` exists on disk. Raises if nothing survives.
6. **Split.** `train_test_split(image_ids, test_size=0.2, random_state=42)`, then `train_test_split(temp_ids, test_size=0.5, random_state=42)`. 80 / 10 / 10 by image, not by box, so no image appears in two splits. Fixed seed, so the split is reproducible.
7. **`VehicleDataset`.** See below.
8. **Transforms.** `Resize((600, 800))` then `ToTensor()`.
9. **DataLoaders.** Batch size 2, `num_workers=0` for Windows, `pin_memory=True`, custom `collate_fn`. Train shuffles; val and test do not.
10. **Model.** `fasterrcnn_resnet50_fpn(weights='DEFAULT')`, then `roi_heads.box_predictor` is swapped for a fresh `FastRCNNPredictor(in_features, num_classes)`. That is the fine-tune: pre-trained backbone and FPN, new classification and regression head sized to this class map.
11. **Optimizer.** `SGD(lr=0.005, momentum=0.9, weight_decay=0.0005)`.
12. **Train loop.** `num_epochs = 2`. `train_epoch` sums the losses in the dict Faster R-CNN returns, backprops, steps, and reports the mean loss per epoch. A `tqdm` bar shows the current batch loss.
13. **Save.** `output_path` is a blank assignment. Set locally. `torch.save(model.state_dict(), output_path)`.

## Class map

`class_to_idx` in the script. Index 0 is reserved for background, which is what torchvision's detection heads expect. 11 foreground classes, 12 entries total, and `num_classes = len(class_to_idx)`.

```
0  background
1  pickup_truck
2  car
3  articulated_truck
4  bus
5  motorized_vehicle
6  work_van
7  single_unit_truck
8  pedestrian
9  bicycle
10 non-motorized_vehicle
11 motorcycle
```

The string `non-motorized_vehicle` uses a hyphen. Class names in `labels.csv` must match these keys exactly or the lookup raises a `KeyError`.

## `VehicleDataset`

A `torch.utils.data.Dataset` over image IDs.

- `__init__` raises if `image_dir` does not exist, so a bad path fails immediately instead of a few hundred batches in.
- `__len__` is the number of image IDs in that split.
- `__getitem__` builds one sample:
  - Missing file on disk: warn and return `(None, None)`.
  - Open as RGB with Pillow.
  - Select the rows for this `image_id`, keep only boxes where `xmax > xmin` and `ymax > ymin`, warn about how many were dropped.
  - No boxes left: emit empty `boxes` and `labels` tensors, which Faster R-CNN accepts as a negative sample.
  - Otherwise build `boxes` as float32 `[N, 4]` and `labels` as int64 `[N]` via `class_to_idx`.
  - Target dict: `boxes`, `labels`, `image_id`, `area`, `iscrowd`. `iscrowd` is all zeros.
  - Apply transforms to the image, return `(image, target)`.
  - Anything that throws is caught, warned, and returned as `(None, None)`.

`collate_fn` filters out the `(None, None)` samples. If a whole batch is empty it returns one dummy 3x224x224 zero image with empty targets, so the loader never hands `[]` to the model.

## Why invalid boxes are dropped

torchvision's Faster R-CNN asserts that every ground-truth box is non-degenerate: `xmax > xmin` and `ymax > ymin`. A box with zero or negative width or height makes the loss undefined and the training run dies mid-epoch with an assertion pointing at a target index, not a filename. The raw labels contain such rows.

So the cleaning happens twice, on purpose:

- **Globally**, right after the CSV load, and it prints the offending rows before removing them. That gives a record of what the data actually contained.
- **Per sample**, inside `__getitem__`, as a guard against any degenerate row that survives filtering or subsetting later.

Dropping the box rather than clamping it is the conservative choice. A zero-area box carries no usable location, and inventing one would be fabricating a label.

## What is not in git

- The image folder. Roughly 5.6k valid JPEGs, matched to labels by an 8-digit filename.
- `labels.csv`.
- `*.pth` trained weights.
- `inferences/`, the visualized detection output.

All four are in `.gitignore`. The repo holds code and docs only. `IMAGE_DIR` and `output_path` are left blank in the committed script rather than pinned to one machine's paths.
