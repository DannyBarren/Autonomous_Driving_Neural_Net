import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import warnings
import time
from tqdm import tqdm  # For progress bar

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(loader, desc="Training batches", leave=False)  # Progress bar
    for images, targets in progress_bar:
        if len(images) == 0:
            continue  # Skip empty batches if any
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        num_batches += 1
        progress_bar.set_postfix(loss=losses.item())  # Show current batch loss
    return total_loss / num_batches if num_batches > 0 else 0.0

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Suppress warnings for clean output (comment out for debugging)
    warnings.filterwarnings("ignore")

    # Define the full image directory path
    IMAGE_DIR = 

    # Loading labels
    labels_df = pd.read_csv('labels.csv', header=None, names=['image_id', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    labels_df['image_id'] = labels_df['image_id'].astype(str).str.zfill(8)  # Ensure 8-digit padding, e.g., '00000156'

    # Validate and clean bounding boxes globally
    labels_df['width'] = labels_df['xmax'] - labels_df['xmin']
    labels_df['height'] = labels_df['ymax'] - labels_df['ymin']
    invalid_boxes = labels_df[(labels_df['width'] <= 0) | (labels_df['height'] <= 0)]
    if not invalid_boxes.empty:
        print(f"\nFound {len(invalid_boxes)} invalid bounding boxes with zero or negative dimensions:")
        print(invalid_boxes[['image_id', 'class', 'xmin', 'ymin', 'xmax', 'ymax']])
        # Remove invalid boxes
        labels_df = labels_df[(labels_df['width'] > 0) & (labels_df['height'] > 0)]
        print(f"Removed invalid boxes. New dataframe size: {len(labels_df)} rows.")

    # Unique image IDs
    image_ids = labels_df['image_id'].unique()

    # Pre-filter valid images
    start_time = time.time()
    valid_image_ids = []
    for img_id in image_ids:
        img_path = os.path.join(IMAGE_DIR, f'{img_id}.jpg')
        if os.path.exists(img_path):
            valid_image_ids.append(img_id)
    print(f"Found {len(valid_image_ids)} valid images out of {len(image_ids)} total IDs. Filtering took {time.time() - start_time:.2f} seconds.")
    image_ids = valid_image_ids  # Replace with only valid ones

    if len(image_ids) == 0:
        raise ValueError("No valid images found. Check your Images.zip extraction or labels.csv.")

    # Split (80% train, 10% val, 10% test)
    train_ids, temp_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    # For testing: Limit to small size to speed up
    # train_ids = train_ids[:100]  # Uncomment to test with 100 images

    # Class mapping
    print("\nClasses in labels dataframe:", labels_df['class'].unique())
    class_to_idx = {  # Assigning numerical IDs to classes for model interpretation
        'background': 0,
        'pickup_truck': 1,
        'car': 2,
        'articulated_truck': 3,
        'bus': 4,
        'motorized_vehicle': 5,
        'work_van': 6,
        'single_unit_truck': 7,
        'pedestrian': 8,
        'bicycle': 9,
        'non-motorized_vehicle': 10,
        'motorcycle': 11
    }
    num_classes = len(class_to_idx)
    print(f"\nNumber of classes in labels dataframe: {num_classes}")

    # Custom PyTorch Dataset with error handling
    class VehicleDataset(Dataset):
        def __init__(self, image_ids, df, image_dir, transforms=None):
            self.image_ids = image_ids
            self.df = df
            self.image_dir = image_dir
            self.transforms = transforms
            # Validate image directory exists
            if not os.path.exists(image_dir):
                raise ValueError(f"Image directory {image_dir} does not exist. Please check your path.")

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            img_path = os.path.join(self.image_dir, f'{img_id}.jpg')  # Assume JPG; adjust if needed
            if not os.path.exists(img_path):
                warnings.warn(f"Image {img_path} not found. Skipping this sample.")
                return None, None  # Return None to skip this sample in DataLoader

            try:
                image = Image.open(img_path).convert('RGB')
                # Filter labels for this image and remove invalid boxes
                annots = self.df[self.df['image_id'] == img_id]
                valid_mask = (annots['xmax'] > annots['xmin']) & (annots['ymax'] > annots['ymin'])
                valid_annots = annots[valid_mask]
                if len(valid_annots) < len(annots):
                    warnings.warn(f"Filtered {len(annots) - len(valid_annots)} invalid boxes for image {img_id}.")

                if len(valid_annots) == 0:
                    # Handle empty annotations: Provide empty tensors (safe for Faster R-CNN)
                    boxes = torch.empty((0, 4), dtype=torch.float32)
                    labels = torch.empty((0,), dtype=torch.int64)
                else:
                    boxes = valid_annots[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float).tolist()  # Ensure float for boxes
                    labels = [class_to_idx[c] for c in valid_annots['class']]
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                    labels = torch.as_tensor(labels, dtype=torch.int64)

                image_id_tensor = torch.tensor([idx])
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.empty((0,))
                iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
                target = {'boxes': boxes, 'labels': labels, 'image_id': image_id_tensor, 'area': area, 'iscrowd': iscrowd}

                if self.transforms:
                    image = self.transforms(image)
                return image, target
            except Exception as e:
                warnings.warn(f"Error processing {img_path}: {str(e)}. Skipping this sample.")
                return None, None

    # Image transformation (normalization for pre-trained models)
    transform = transforms.Compose([transforms.Resize((600, 800)), transforms.ToTensor()])
    print('\nImage Folder Path Located:', IMAGE_DIR)

    # Create datasets
    train_dataset = VehicleDataset(train_ids, labels_df, IMAGE_DIR, transform)
    val_dataset = VehicleDataset(val_ids, labels_df, IMAGE_DIR, transform)
    test_dataset = VehicleDataset(test_ids, labels_df, IMAGE_DIR, transform)

    # Custom collate function to handle None returns
    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]  # Filter out None samples
        if len(batch) == 0:
            # Dummy with empty targets to avoid empty batch error
            dummy_image = torch.zeros(3, 224, 224)  # Dummy image tensor (C, H, W)
            dummy_target = {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64),
                'image_id': torch.tensor([0]),
                'area': torch.empty((0,)),
                'iscrowd': torch.empty((0,), dtype=torch.int64)
            }
            return [dummy_image], [dummy_target]
        return tuple(zip(*batch))

    # DataLoaders with num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torch.optim import SGD
    import torch

    # Load pre-trained Faster R-CNN and fine-tune for your classes
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Device (GPU if available)
    model.to(device)

    # Optimizer (SGD with corrected momentum)
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 2  # Reduced for testing
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}: Loss = {train_loss:.4f}')

    # Saving model
    output_path = 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)

    print("\nTraining Complete.")
