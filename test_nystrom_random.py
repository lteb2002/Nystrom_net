import torch
from torch import optim
import dl_model.rere_tsne as tsne
import numpy as np
import dl_model.nystrom_net as m
import dl_model.rere_config as cnf
import dl_model.dl_helper as hl
import dl_model.rere_nystrom as nys
import time
from torchtools.optim import RangerLars

batch_size = 1000
epoch_num = 20
kernel_name = 'rbf'
if_fix_nys = False
if_dml_reg = False
sample_size = 100

fp = 'H:\\nystrom_experiment\\'
fn = 'gao_xin_numeric_n2b'
fnys = fn + '_nys_fix' if if_fix_nys else fn + '_nys_rand'
fdml = fn + '_dml_fix' if if_fix_nys else fn + '_dml_rand'
input_file = fp + fn + '.arff'

output1 = fp + fnys + '.csv'
output2 = fp + fdml + '.csv'
img1 = fp + 'images\\' + fn + '.png'
img2 = fp + 'images\\' + fnys + '.png'
img3 = fp + 'images\\' + fdml + '.png'
model_path = fp + 'torch_models\\' + fnys

data_set = hl.ArffDataSet(input_file, normalize=True)
train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)

labels = data_set.labels.cpu().numpy()
viz = tsne.RereTSNE(data_set.data.cpu().numpy(), labels)
viz.save_image(img1)

d_in = data_set.dim
print("Dimension of the data:", d_in)
d_out = data_set.label_num
print("class number:", d_out)

rows = data_set.data.shape[0]
inds = np.random.choice(range(rows), sample_size, replace=False)
samples = data_set.data[inds]
print("Sample size in Nystrom:", samples.shape[0])
model = m.NystromNet(d_in, samples, d_out, if_dml_reg, if_fix_nys, kernel_function=kernel_name).to(cnf.device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# optimizer = RangerLars(model.parameters())


def train(epoch):
    hl.train(epoch, model, optimizer, train_loader)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(cnf.device)
            recon_batch, mu, logvar = model(data)
            test_loss += m.loss_function(data, recon_batch, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def transform_save():
    # 保存训练好的模型参数
    torch.save(model.state_dict(), model_path)
    # 变换出Nystrom数据
    nys_data = model.transform(data_set.data.to(cnf.device)).detach().cpu().numpy()
    # 变换出DML数据
    dml_data = model.encode_dml(data_set.data.to(cnf.device)).detach().cpu().numpy()
    # 可视化Nystrom数据
    viz1 = tsne.RereTSNE(nys_data, labels)
    viz1.save_image(img2)
    # 可视化DML数据
    viz2 = tsne.RereTSNE(dml_data, labels)
    viz2.save_image(img3)
    # 保存Nystrom变换后的数据
    hl.save_numpy_data_to_csv(nys_data, labels, output1)
    # 保存DML变换后的数据
    hl.save_numpy_data_to_csv(dml_data, labels, output2)


if __name__ == "__main__":
    begin = time.time()
    for epoch in range(1, epoch_num + 1):
        train(epoch)
    transform_save()
    end = time.time()
    print(fn, ' for dataset ', fn, ' time cost:', end - begin)
    # for p in model.named_parameters():
    #     print(p)
