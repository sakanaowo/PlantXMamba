import os
import re
import matplotlib.pyplot as plt

# Đọc file log
with open('./utils/log_text.txt', 'r', encoding='utf-8') as f:
    log_text = f.read()

log_name = input("Nhập tên file log (mặc định là 'log_text.txt'): ")

# Trích xuất dữ liệu bằng regex
pattern = r'Epoch \[\d+/\d+\] Train Loss: ([\d\.]+), Acc: ([\d\.]+) \| Val Loss: ([\d\.]+), Acc: ([\d\.]+)'
lines = re.findall(pattern, log_text)

# Tách ra các list số
train_losses = [float(l[0]) for l in lines]
train_accuracies = [float(l[1]) for l in lines]
val_losses = [float(l[2]) for l in lines]
val_accuracies = [float(l[3]) for l in lines]

# Tạo thư mục lưu trữ nếu chưa tồn tại
save_dir = './outputs/plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Tạo thư mục 'plots'

# Tạo epochs
epochs = range(1, len(train_losses) + 1)

# Biểu đồ Loss
plt.figure(figsize=(14, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.title(log_name + ': Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, log_name + '_loss.png'), dpi=300, bbox_inches='tight')  # Lưu biểu đồ Loss
plt.close()  # Đóng biểu đồ

# Biểu đồ Accuracy
plt.figure(figsize=(14, 6))
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.title(log_name + ': Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, log_name + '_accuracy.png'), dpi=300, bbox_inches='tight')  # Lưu biểu đồ Accuracy
plt.close()  # Đóng biểu đồ