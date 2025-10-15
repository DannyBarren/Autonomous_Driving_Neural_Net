# Object Detection Neural Network for Autonomous Vehicles

## Introduction
This neural network is a key component of my capstone project for Simplilearn's Caltech AI/ML Programming certification course. The model is an object detection system designed to identify and classify various vehicle types using a labeled dataset comprising over 5,000 images. Due to size constraints, I have not included the dataset in this repository, but I can provide it as a .zip file upon request. The dataset consists of two primary files: a folder containing more than 5,000 images named with unique ID numbers (ranging from 0000000 to 005673), and a corresponding .csv file that lists each ID, the vehicle type label (e.g., motorcycle, truck), and the coordinates of a bounding box pinpointing the vehicle’s location within each image, enabling precise localization by the model. This project posed several challenges, particularly in redesigning the data pipeline to optimize performance on the hardware available to me at the time—my personal laptop. Through iterative adjustments, I successfully adapted the model to run efficiently, trained it effectively, and saved the resulting model weights for future use. 

## Project Details
### Objective
The primary goal was to develop an AI model capable of predicting vehicle types (e.g., cars, trucks, pedestrians) and localizing them with bounding boxes, aligning with the requirements of autonomous vehicle technology. This project leverages a dataset of images and corresponding labels, processed to train a robust detection system.

### Technical Implementation
- **Framework**: Utilizes PyTorch with a pre-trained Faster R-CNN model based on ResNet50.
- **Dataset**: Processed 5,626 valid images from an initial 110,000, split into 4,500 training, 563 validation, and 563 test images. Data cleaning removed invalid bounding boxes, ensuring high-quality input.
- **Training**: Conducted over two epochs on an NVIDIA GeForce GTX 1650 GPU, achieving a training loss reduction from 0.4989 to 0.4528, demonstrating effective learning.
- **Features**: Includes data loading, preprocessing (resizing to 600x800 pixels), custom dataset class, and visualization of inference results with bounding boxes.

### Code Structure
- `Barren_TeslaCapstone_Part1.py`: Main script containing data handling, model training, and inference logic.
- `images/`: Directory for the image dataset (placeholder; actual data not included due to size).
- `trained_model.pth`: Saved model weights post-training.
- `inferences/`: Folder for visualized inference outputs (generated during execution).

## Value to Employers
This project highlights my proficiency in deep learning and object detection, skills directly applicable to industries like automotive, robotics, and surveillance. My ability to preprocess large datasets, optimize neural network performance, and deploy GPU-accelerated solutions can enhance real-time systems, reduce development costs, and improve safety features in autonomous vehicles. The code’s modular design and error handling reflect my commitment to robust, production-ready software, making me a valuable asset for teams tackling complex AI challenges.

## How to Use
1. **Setup**: Ensure PyTorch, torchvision, pandas, PIL, and matplotlib are installed (`pip install torch torchvision pandas pillow matplotlib`).
2. **Data Preparation**: Place your image dataset in the `images/` folder and update `labels.csv` with image IDs, classes, and bounding box coordinates.
3. **Run the Script**: Execute `Barren_TeslaCapstone_Part1.py` to train the model and generate inferences. Adjust `IMAGE_DIR` and `MODEL_OUTPUT_PATH` in the code to match your environment.
4. **Output**: Check `inferences/` for visualized detection results and `trained_model.pth` for the trained model.

## Results
Initial training shows promising loss reduction, with potential for further optimization through additional epochs. Inference visualizations confirm the model’s ability to detect vehicles, though dataset completeness should be verified for full accuracy.

## Skills Demonstrated
- Deep Learning with PyTorch
- Object Detection (Faster R-CNN)
- Data Preprocessing and Cleaning
- GPU Computing
- Model Evaluation and Visualization

## Future Improvements
I plan to expand the dataset, increase training epochs for better convergence, and integrate advanced metrics like mean Average Precision (mAP) to enhance model performance.

## Last Updated
07:10 PM EDT on Wednesday, October 15, 2025

This project is a testament to my hands-on experience in AI, and I am eager to apply these skills to innovative solutions in your organization. Explore the code, and feel free to reach out for collaboration or discussion!
