import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from Src.SINet import SINet_ResNet101  # Ganti dengan SINet_ResNet101
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor

# Definisi fungsi untuk metrik evaluasi
def compute_e_measure(cam_pred, cam_gt):
    return np.sum(2 * cam_pred * cam_gt) / (np.sum(cam_pred) + np.sum(cam_gt) + 1e-8)

def compute_s_measure(cam_pred, cam_gt):
    mean_p = np.mean(cam_pred)
    mean_g = np.mean(cam_gt)
    cov_pg = np.cov(cam_pred, cam_gt)[0, 1]
    return (2 * cov_pg + 1e-8) / (mean_p**2 + mean_g**2 + 1e-8)

def compute_weighted_f_measure(cam_pred, cam_gt):
    e_measure = compute_e_measure(cam_pred, cam_gt)
    s_measure = compute_s_measure(cam_pred, cam_gt)
    return (1 + s_measure) * e_measure / (s_measure + e_measure + 1e-8)

# Argumen dari command line untuk pengaturan
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str, default='./Snapshot/2020-CVPR-SINet/SINet_40.pth')
parser.add_argument('--test_save', type=str, default='./Result/2020-CVPR-SINet-New/')
parser.add_argument('--result_file', type=str, default='./Result/2020-CVPR-SINet-New/results.csv')
opt = parser.parse_args()

# Load model SINet_ResNet101 dan pindahkan ke GPU
model = SINet_ResNet101().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

# Loop untuk setiap dataset yang diuji, misalnya 'COD10K'
results = []


for dataset in ['COD10K', 'CAMO', 'CHAMELEON']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    
    # Load dataset uji dengan ukuran yang sesuai
    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Imgs/'.format(dataset),
                               gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    
    # Loop untuk setiap iterasi pada dataset uji
    for iteration in range(test_loader.size):
        # Load data gambar, ground-truth, dan nama
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        # Inferensi menggunakan model
        _, cam = model(image)
        
        # Reshape dan squeeze hasil inferensi
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        
        # Normalisasi hasil
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Simpan gambar menggunakan matplotlib
        plt.imsave(save_path + name, cam, cmap='viridis')  # Misalnya menggunakan cmap 'viridis'
        
        # Evaluasi MAE
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        
        # Evaluasi E-measure, S-measure, weighted F-measure
        e_measure = compute_e_measure(cam, gt)
        s_measure = compute_s_measure(cam, gt)
        weighted_f_measure = compute_weighted_f_measure(cam, gt)
        
        # Simpan hasil evaluasi dalam list
        results.append({
            'Dataset': dataset,
            'Image': name,
            'MAE': mae,
            'E-measure': e_measure,
            'S-measure': s_measure,
            'Weighted F-measure': weighted_f_measure
        })
        
        # Cetak hasil evaluasi
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}, E-measure: {}, S-measure: {}, Weighted F-measure: {}'.format(
            dataset, name, img_count, test_loader.size, mae, e_measure, s_measure, weighted_f_measure))
        
        img_count += 1


# Konversi hasil evaluasi menjadi DataFrame pandas
df_results = pd.DataFrame(results)

# Simpan hasil evaluasi ke dalam file CSV
df_results.to_csv(opt.result_file, index=False)

# Tampilkan DataFrame
print("\nResults Table:")
print(df_results)

print("\n[Congratulations! Testing Done]")
