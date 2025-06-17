# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = PlantXViT(num_classes=4).to(device)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
# history = train_model(
#     model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     criterion=criterion,
#     optimizer=optimizer,
#     num_epochs=50,
#     device=device,
#     save_path=config["output"]["model_path"]
# )

import torch
from mamba_ssm import Mamba2

model = Mamba2(d_model=16, d_state=64, d_conv=4, expand=2)
x = torch.randn(1, 100, 16)
output = model(x)
print(output.shape)  # Mong đợi: torch.Size([1, 100, 16])