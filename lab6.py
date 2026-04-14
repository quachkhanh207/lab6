import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Tên file ảnh của bạn (phải để cùng thư mục với file code này)
IMAGE_NAME = 'test.jpg' 

def run_lab():
    # Thử đọc ảnh
    img = cv2.imread(IMAGE_NAME)
    
    # Nếu không có ảnh thật, code tự tạo 1 ảnh màu để bạn nộp bài không bị lỗi
    if img is None:
        print(f"Canh bao: Khong tim thay file {IMAGE_NAME}. Dang tao anh mau de test!")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(img, 'Demo Image', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    # --- HIỂN THỊ KẾT QUẢ ---
    plt.figure(figsize=(15, 5))
    
    # 1. Ảnh gốc
    plt.subplot(1, 5, 1)
    plt.imshow(img_resized)
    plt.title("Original (224x224)")
    plt.axis('off')

    # 2. Tạo 4 bản Augmentation
    for i in range(2, 6):
        # Augmentation: Xoay ngẫu nhiên
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
        aug = cv2.warpAffine(img_resized, M, (224, 224))
        
        # Augmentation: Lật ngang
        if random.random() > 0.5:
            aug = cv2.flip(aug, 1)
            
        # Chuẩn hóa pixel về [0, 1]
        aug_normalized = aug.astype(np.float32) / 255.0
        
        plt.subplot(1, 5, i)
        plt.imshow(aug_normalized)
        plt.title(f"Augmented {i-1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lab()