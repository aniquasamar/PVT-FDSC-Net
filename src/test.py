import sys
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.pvt_fdsc_net import PVTFDSCNet
from utils import create_dir, seeding
from utils import calculate_metrics
from metrics import hd_dist

def calculate_hd(y_true, y_pred):
    # Standardizing shapes for Hausdorff Distance
    y_true = y_true[0][0].detach().cpu().numpy()
    y_pred = y_pred[0][0].detach().cpu().numpy()

    y_pred = (y_pred > 0.5).astype(np.uint8)
    y_true = (y_true > 0.5).astype(np.uint8)

    return hd_dist(y_true, y_pred)

def evaluate(model, save_path, test_x, test_y, size):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Adjusted for 7 metrics
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image Processing """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image # for joint save
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).to(device, dtype=torch.float32)

        """ Mask Processing """
        mask_img = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, size)
        save_mask = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB) # for joint save
        
        mask = mask_img/255.0
        mask = np.expand_dims(mask, axis=(0, 1))
        mask = torch.from_numpy(mask).to(device, dtype=torch.float32)

        with torch.no_grad():
            """ Inference """
            start_time = time.time()
            y_final = model(image) 
            
            # 2. Apply sigmoid to the final prediction head
            y_pred = torch.sigmoid(y_final)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Metrics """
            score = calculate_metrics(mask, y_pred) # Returns [jac, f1, recall, prec, acc, f2]
            hd = calculate_hd(mask, y_pred)
            score.append(hd)
            metrics_score = list(map(add, metrics_score, score))

            """ Post-processing Predicted Mask """
            p = y_pred[0][0].cpu().numpy()
            p = (p > 0.5).astype(np.uint8) * 255
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)

        """ Visual Saving """
        line = np.ones((size[1], 10, 3)) * 255 # Vertical white separator
        cat_images = np.concatenate([save_img, line, save_mask, line, p], axis=1)
        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", p)

    # Calculate final averages
    samples = len(test_x)
    jaccard, f1, recall, precision, acc, f2, hd = [s/samples for s in metrics_score]
    fps = 1 / np.mean(time_taken)

    print(f"Jaccard: {jaccard:.4f} - F1: {f1:.4f} - Recall: {recall:.4f} - Precision: {precision:.4f} - Acc: {acc:.4f} - F2: {f2:.4f} - HD: {hd:.4f} - FPS: {fps:.2f}")
    return [jaccard, f1, recall, precision, acc, f2, hd, fps]

if __name__ == "__main__":
    seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ Load Model """
    model = PVTFDSCNet().to(device)
    checkpoint_path = "files/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Weights loaded successfully.")
    
    model.eval() # This triggers the toggle to return only 1 output

    datasets = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
    base_path = "/content/TestDataset_Temp/TestDataset" 

    for ds in datasets:
        dataset_path = os.path.join(base_path, ds)
        test_x = sorted(glob(os.path.join(dataset_path, "images", "*.*")))
        test_y = sorted(glob(os.path.join(dataset_path, "masks", "*.*")))

        if len(test_x) == 0: continue

        save_path = f"resultsfiles/{ds}"
        create_dir(os.path.join(save_path, "mask"))
        create_dir(os.path.join(save_path, "joint"))

        print(f"\n--- Evaluation on {ds} ---")
        evaluate(model, save_path, test_x, test_y, size=(352, 352))