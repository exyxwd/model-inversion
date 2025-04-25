# Filename: phase2_train_attack_model.py
# --- Start of generated code ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Định nghĩa Attack Model phức tạp hơn ---
class AttackModel(nn.Module):
    def __init__(self, input_dim=10, latent_dim=100, img_channels=1, img_size=28):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 4 # Kích thước ban đầu sau khi reshape (ví dụ: 28 // 4 = 7)

        # Lớp linear để mở rộng vector đầu vào thành không gian tiềm ẩn lớn hơn
        self.fc_expand = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
             nn.Linear(latent_dim, 128 * self.init_size * self.init_size), # Mở rộng đủ lớn để reshape
             nn.BatchNorm1d(128 * self.init_size * self.init_size),
             nn.LeakyReLU(0.2, inplace=True)
        )

        # Các lớp ConvTranspose2d để xây dựng lại ảnh
        self.decoder = nn.Sequential(
            # Input: [batch, 128, init_size, init_size] (ví dụ: [batch, 128, 7, 7])
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Output: [batch, 64, init_size*2, init_size*2] (ví dụ: 14x14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1), # Output: [batch, C, init_size*4, init_size*4] (ví dụ: 28x28)
            nn.Tanh() # Sử dụng Tanh (-1 đến 1) hoặc Sigmoid (0 đến 1) tùy thuộc vào cách chuẩn hóa ảnh gốc
                      # MNIST gốc thường được chuẩn hóa [-1, 1] hoặc [0, 1]. ToTensor() chuẩn hóa về [0, 1] nên Sigmoid phù hợp hơn.
            # nn.Sigmoid() # Dùng Sigmoid nếu ảnh gốc là [0, 1]
        )
        # Điều chỉnh: Nếu dùng Sigmoid thì thay Tanh() bằng Sigmoid()
        # Sửa lại decoder cuối cùng để dùng Sigmoid cho MNIST [0,1]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Phù hợp với ảnh MNIST chuẩn hóa [0, 1] bởi ToTensor()
        )


    def forward(self, x):
        # x có shape [batch, input_dim]
        out = self.fc_expand(x)
        # Reshape để phù hợp với ConvTranspose2d
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        # Decode thành ảnh
        img = self.decoder(out)
        # img có shape [batch, img_channels, img_size, img_size]
        return img

# --- Hàm huấn luyện giữ nguyên logic nhưng dùng AttackModel mới ---
def train_attack_model(vectors, images, epochs=100, batch_size=32, lr=0.00001, device='cpu'): # Giảm LR một chút cho mô hình phức tạp hơn
    """
    Trains the Attack Model (Generator-like).

    Args:
        vectors (Tensor): Confidence vectors (input).
        images (Tensor): Ground truth images (target).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        device (str): Device ('cuda' or 'cpu').

    Returns:
        AttackModel: The trained attack model.
    """
    input_dim = vectors.shape[1]
    img_channels = images.shape[1]
    img_size = images.shape[2] # Giả sử ảnh vuông

    model = AttackModel(input_dim=input_dim, img_channels=img_channels, img_size=img_size).to(device)
    model.train() # Set model to training mode

    # Chuẩn bị DataLoader
    dataset = TensorDataset(vectors, images) # Input là vectors, target là images
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Thử lr = 0.00001 (1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.5, 0.999))

    #optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999)) # Dùng betas thường thấy trong GAN training
    loss_fn = nn.L1Loss() # Vẫn dùng MSE hoặc có thể thử L1Loss (nn.L1Loss())

    print(f" Training Attack Model ({epochs} epochs, lr={lr})...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (batch_vectors, batch_images) in enumerate(loader):
            batch_vectors = batch_vectors.to(device)
            batch_images = batch_images.to(device)

            optimizer.zero_grad()
            # Tạo ảnh từ vectors
            reconstructed_images = model(batch_vectors)
            # Tính loss
            loss = loss_fn(reconstructed_images, batch_images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_vectors.size(0)

        avg_epoch_loss = epoch_loss / len(dataset)
        # In loss định kỳ để theo dõi
        if (epoch + 1) % 50 == 0: # In mỗi 50 epochs
             print(f"  Attack Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.6f}")

    # In loss cuối cùng
    print(f"  Final Attack Model Average Loss: {avg_epoch_loss:.6f}")
    model.eval() # Chuyển sang eval mode
    return model.to('cpu') # Trả model về CPU sau khi train

# --- End of generated code ---