# # Giả sử log_text chứa toàn bộ đoạn log bạn dán bên trên (dạng string)
# from utils.helper import parse_log_file
#
# train_loss, train_acc, val_loss, val_acc = parse_log_file('../outputs/embrapa/logs/training_attempt2.txt')
#
# # Vẽ biểu đồ
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 5))
#
# # Accuracy
# # plt.subplot(1, 2, 1)
# plt.plot(train_acc, label='Train Acc')
# plt.plot(val_acc, label='Val Acc')
# plt.title('Accuracy per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()
#
# # Loss
# # plt.subplot(1, 2, 2)
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Val Loss')
# plt.title('Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

import re

log_text="""
Epoch 1/50
Training: 100%|██████████| 1851/1851 [03:22<00:00,  9.12it/s]
Evaluating: 100%|██████████| 466/466 [00:21<00:00, 22.16it/s]
Train Loss: 3.3308 | Acc: 0.3331
Val   Loss: 2.6582 | Acc: 0.4710
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 2/50
Training: 100%|██████████| 1851/1851 [03:23<00:00,  9.11it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.44it/s]
Train Loss: 2.3098 | Acc: 0.4958
Val   Loss: 1.9044 | Acc: 0.5836
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 3/50
Training: 100%|██████████| 1851/1851 [03:22<00:00,  9.14it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.57it/s]
Train Loss: 1.7258 | Acc: 0.6038
Val   Loss: 1.4550 | Acc: 0.6708
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 4/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.18it/s]
Evaluating: 100%|██████████| 466/466 [00:21<00:00, 22.19it/s]
Train Loss: 1.3461 | Acc: 0.6878
Val   Loss: 1.1953 | Acc: 0.7172
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 5/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.19it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.29it/s]
Train Loss: 1.1103 | Acc: 0.7336
Val   Loss: 1.0832 | Acc: 0.7262
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 6/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.18it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.42it/s]
Train Loss: 0.9340 | Acc: 0.7680
Val   Loss: 0.8805 | Acc: 0.7832
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 7/50
Training: 100%|██████████| 1851/1851 [03:22<00:00,  9.13it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.28it/s]
Train Loss: 0.8183 | Acc: 0.7936
Val   Loss: 0.8044 | Acc: 0.7922
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 8/50
Training: 100%|██████████| 1851/1851 [03:23<00:00,  9.11it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.32it/s]
Train Loss: 0.7231 | Acc: 0.8144
Val   Loss: 0.7011 | Acc: 0.8216
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 9/50
Training: 100%|██████████| 1851/1851 [03:22<00:00,  9.16it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.41it/s]
Train Loss: 0.6485 | Acc: 0.8307
Val   Loss: 0.6582 | Acc: 0.8252
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 10/50
Training: 100%|██████████| 1851/1851 [03:23<00:00,  9.12it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.56it/s]
Train Loss: 0.5884 | Acc: 0.8430
Val   Loss: 0.6312 | Acc: 0.8333
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 11/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.18it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.55it/s]
Train Loss: 0.5279 | Acc: 0.8575
Val   Loss: 0.6260 | Acc: 0.8270

Epoch 12/50
Training: 100%|██████████| 1851/1851 [03:23<00:00,  9.10it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.45it/s]
Train Loss: 0.4913 | Acc: 0.8674
Val   Loss: 0.5763 | Acc: 0.8381
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 13/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.18it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.44it/s]
Train Loss: 0.4460 | Acc: 0.8786
Val   Loss: 0.5534 | Acc: 0.8451
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 14/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.19it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.61it/s]
Train Loss: 0.4119 | Acc: 0.8862
Val   Loss: 0.5185 | Acc: 0.8551
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 15/50
Training: 100%|██████████| 1851/1851 [03:22<00:00,  9.14it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.65it/s]
Train Loss: 0.3831 | Acc: 0.8926
Val   Loss: 0.5331 | Acc: 0.8453

Epoch 16/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.18it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.60it/s]
Train Loss: 0.3523 | Acc: 0.9019
Val   Loss: 0.4912 | Acc: 0.8575
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 17/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.18it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.49it/s]
Train Loss: 0.3318 | Acc: 0.9063
Val   Loss: 0.4635 | Acc: 0.8691
✅ Saved best model to ./outputs/embrapa/models/plantxvit_best.pth

Epoch 18/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.17it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.62it/s]
Train Loss: 0.3069 | Acc: 0.9131
Val   Loss: 0.4839 | Acc: 0.8596

Epoch 19/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.21it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.75it/s]
Train Loss: 0.2888 | Acc: 0.9182
Val   Loss: 0.4778 | Acc: 0.8648

Epoch 20/50
Training: 100%|██████████| 1851/1851 [03:22<00:00,  9.12it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.58it/s]
Train Loss: 0.2719 | Acc: 0.9225
Val   Loss: 0.4925 | Acc: 0.8585

Epoch 21/50
Training: 100%|██████████| 1851/1851 [03:21<00:00,  9.20it/s]
Evaluating: 100%|██████████| 466/466 [00:20<00:00, 22.71it/s]
Train Loss: 0.2598 | Acc: 0.9273
Val   Loss: 0.4744 | Acc: 0.8652
"""

# Tìm tất cả các dòng Train và Val
train_lines = re.findall(r'Train Loss: ([\d\.]+) \| Acc: ([\d\.]+)', log_text)
val_lines = re.findall(r'Val\s+Loss: ([\d\.]+) \| Acc: ([\d\.]+)', log_text)

# Tách ra các list số
train_losses = [float(l[0]) for l in train_lines]
train_accuracies = [float(l[1]) for l in train_lines]
val_losses = [float(l[0]) for l in val_lines]
val_accuracies = [float(l[1]) for l in val_lines]

import matplotlib.pyplot as plt

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(14, 6))

# Loss
# plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.title('Embrapa: Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Accuracy
# plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.title('Embrapa: Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# import os
# import shutil
# from sklearn.model_selection import train_test_split
#
# # Đường dẫn đến thư mục raw
# raw_dir = '/home/sakana/Code/PlantVillage-Dataset/raw/color'
# output_base_dir = '/home/sakana/Code/PlantXViT/data/raw/plantvillage'
#
# # Tạo các thư mục mới nếu chưa tồn tại
# for split in ['train', 'val', 'test']:
#     os.makedirs(os.path.join(output_base_dir, split), exist_ok=True)
#
# # Lấy danh sách các thư mục nhãn trong raw
# labels = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
#
# # Duyệt qua từng nhãn
# for label in labels:
#     label_dir = os.path.join(raw_dir, label)
#     # Kiểm tra tất cả định dạng ảnh phổ biến
#     image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG')
#     images = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(image_extensions)]
#
#     # Kiểm tra và in số lượng ảnh thực tế
#     if len(images) == 0:
#         print(f"Không tìm thấy ảnh hợp lệ trong thư mục {label}. Kiểm tra định dạng hoặc tệp hỏng.")
#         # Thử liệt kê tất cả tệp để kiểm tra
#         all_files = os.listdir(label_dir)
#         if all_files:
#             print(f"Tệp trong {label}: {all_files}")
#         continue
#
#     # Xử lý trường hợp số lượng ảnh nhỏ
#     if len(images) < 2:
#         print(f"Chỉ có {len(images)} ảnh trong thư mục {label}. Đặt tất cả vào train.")
#         split_label_dir = os.path.join(output_base_dir, 'train', label)
#         os.makedirs(split_label_dir, exist_ok=True)
#         for img_path in images:
#             try:
#                 destination = os.path.join(split_label_dir, os.path.basename(img_path))
#                 shutil.copy(img_path, destination)
#             except Exception as e:
#                 print(f"Lỗi sao chép {img_path}: {e}")
#         continue
#     elif len(images) < 5:
#         print(f"Ít ảnh trong {label} ({len(images)}). Giảm test_size.")
#         train_images, val_test_images = train_test_split(images, test_size=0.2, random_state=42, stratify=[label] * len(images))
#         val_images, test_images = train_test_split(val_test_images, test_size=0.5, random_state=42, stratify=[label] * len(val_test_images))
#     else:
#         train_images, val_test_images = train_test_split(images, test_size=0.3, random_state=42, stratify=[label] * len(images))
#         val_images, test_images = train_test_split(val_test_images, test_size=0.5, random_state=42, stratify=[label] * len(val_test_images))
#
#     # Tạo thư mục nhãn trong train, val, test
#     for split, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
#         split_label_dir = os.path.join(output_base_dir, split, label)
#         os.makedirs(split_label_dir, exist_ok=True)
#
#         # Di chuyển ảnh vào thư mục tương ứng
#         for img_path in split_images:
#             try:
#                 destination = os.path.join(split_label_dir, os.path.basename(img_path))
#                 shutil.copy(img_path, destination)
#             except Exception as e:
#                 print(f"Lỗi sao chép {img_path}: {e}")
#
#     print(f"Phân chia xong cho nhãn {label}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
#
# # Tổng số ảnh
# total_images = sum(len([f for f in os.listdir(os.path.join(raw_dir, label)) if f.endswith(image_extensions)]) for label in labels)
# print(f"Tổng số ảnh hợp lệ: {total_images}")
# print(f"Số ảnh dự kiến - Train: {int(total_images * 0.7)}, Val: {int(total_images * 0.15)}, Test: {int(total_images * 0.15)}")