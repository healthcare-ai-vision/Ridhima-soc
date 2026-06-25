# Week 3–4: Plant Disease Classification with Roboflow Universe

This project trains an image classification model using transfer learning
(ResNet18 pretrained on ImageNet) to detect plant diseases from photos of
citrus fruit and leaves, using a dataset sourced from
[Roboflow Universe](https://universe.roboflow.com/).

## Dataset

**[Plant Diseases](https://universe.roboflow.com/plant-diseases/plant-diseases-oqkrq)**
(hosted on Roboflow Universe, published by "Plant Diseases")

- **Task:** Image Classification
- **Classes (10):**
  - `black_spot_fruit`, `black_spot_leave`
  - `canker_fruit`, `canker_leave`
  - `greening__fruit`, `greening_leave`
  - `healthy_fruit`, `healthy_leave`
  - `melanose_leave`
  - `scab_fruit`
- **Size:** 963 images
- **License:** Public Domain (CC0)
- **Why this dataset:** plant disease detection is a practical, easy-to-explain
  computer vision use case (smart agriculture / crop monitoring), the classes
  are visually distinct (different blemish patterns on fruit vs. leaves), and
  it's small enough to train quickly on a laptop or free Colab GPU while still
  being a genuine multi-class (10-class) classification problem.

> Note: with only ~963 images across 10 classes, this is a small dataset.
> Transfer learning (reusing ImageNet-pretrained features, only training a
> new classifier head) is what makes this feasible — training a CNN from
> scratch on this little data would overfit badly. See "Approach notes" below.

> Want to swap datasets? Any classification dataset on Roboflow Universe works.
> Just open its page, click **Download Dataset → Folder Structure**, and copy
> the `workspace` / `project` / `version` values into
> `src/download_dataset.py`.

## Project structure

```
week3-4/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # downloaded dataset goes here (gitignored)
├── outputs/               # trained model + plots + metrics (gitignored)
└── src/
    ├── download_dataset.py   # pulls the dataset from Roboflow
    ├── train.py               # trains the classifier
    └── predict.py             # runs inference on new images
```

## Setup

```bash
git clone <your-repo-url>
cd week3-4
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Download the dataset

Get a free Roboflow API key from <https://app.roboflow.com/settings/api>,
then run:

```bash
export ROBOFLOW_API_KEY="your_key_here"
python src/download_dataset.py
```

This downloads the dataset into `data/` in a standard
`train/<class>/*.jpg` folder layout.

## 2. Train the model

```bash
python src/train.py --data_dir data --output_dir outputs --epochs 15 --batch_size 16
```

A smaller `--batch_size` (e.g. 16) is recommended here since the dataset is
small — it gives more gradient updates per epoch.

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10 | number of training epochs |
| `--batch_size` | 32 | batch size |
| `--lr` | 1e-3 | learning rate |
| `--img_size` | 224 | input image resolution |
| `--unfreeze_backbone` | off | fine-tune the full ResNet18 instead of just the classifier head (slower, usually more accurate, but more prone to overfitting on this small dataset) |

This will:
- fine-tune a ResNet18 head on the dataset
- print per-epoch train/validation loss and accuracy
- save the best checkpoint to `outputs/model.pt`
- save a loss/accuracy plot to `outputs/training_curves.png`
- save per-epoch metrics to `outputs/history.json`
- evaluate on the `test` split (if present) and save `outputs/test_metrics.json`

## 3. Run inference

```bash
python src/predict.py --model outputs/model.pt --image path/to/leaf_photo.jpg
python src/predict.py --model outputs/model.pt --image_dir path/to/folder_of_photos
```

Prints the top-3 predicted classes (e.g. `canker_leave`, `healthy_leave`,
`scab_fruit`) with confidence scores for each image.

## Approach notes

- **Backbone:** ResNet18 pretrained on ImageNet (`torchvision.models`).
- **Transfer learning:** by default only the final fully-connected layer is
  trained (backbone frozen). This matters a lot here — with under 1,000
  images, fine-tuning the whole network risks overfitting fast. Pass
  `--unfreeze_backbone` only if you also lower the learning rate and watch
  validation accuracy closely.
- **Augmentation:** random horizontal flip, small rotations, and color jitter
  on the training set only — important for a small dataset, since it
  effectively multiplies the variety of training examples the model sees.
- **Validation:** uses the dataset's own `valid` split when available;
  otherwise carves off 15% of the training set automatically.
- **Class imbalance:** the 10 classes aren't perfectly balanced (some disease
  categories have more fruit images than leaf images, etc.). If validation
  accuracy looks misleadingly high, check `outputs/test_metrics.json` and
  consider inspecting per-class performance, since accuracy alone can hide
  poor performance on minority classes.

## Results

After training, check `outputs/training_curves.png` and
`outputs/test_metrics.json` for final accuracy numbers. *(Fill in your actual
numbers here once you've run training, e.g. "Achieved 82% validation accuracy
after 15 epochs.")*
