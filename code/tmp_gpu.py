import torch
for device_id in range(8):
    device = 'cuda:%d'%device_id
    _, __ = torch.cuda.mem_get_info(device=device)
    print(device_id, 'Free:', _/1024/1024, 'Total:', __/1024/1024)