import torch
from torch import optim
import dl_model.svdd.rere_svdd as m
import dl_model.svdd.rere_svdd_trainer as hl
import dl_model.rere_tsne as tsne
import dl_model.rere_config as cnf
import numpy as np

batch_size = 2000
epoch_num = 30

fp = 'H:\\svdd_experiments\\'
fn = 'bank_marketing'
fdml = fn + '_svdd'
input_file = fp + fn + '.arff'

output2 = fp + fdml + '.csv'
img1 = fp + 'images\\' + fn + '.png'
img3 = fp + 'images\\' + fdml + '.png'
model_path = fp + 'torch_models\\' + fdml
log_file = fp + 'torch_models\\' + fdml+'_log.csv'

# fp = 'H:\\url_experiments\\url_sample_50000.arff'
# output = 'H:\\url_experiments\\y_urls_sample_50000_nor_vae.csv'
data_set = hl.ArffDataSet(input_file)
train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

labels = data_set.labels.cpu().numpy()
viz = tsne.RereTSNE(data_set.data.cpu().numpy(), labels)
viz.save_image(img1)

ls = np.unique(labels)
print(ls)
print(len(labels[labels == 0]))

# 正常类别的标签
max_l = 0
max_num = 0
for i in range(0, len(ls)):
    num = len(labels[labels == i])
    if num > max_num:
        max_l = i
        max_num = num
print("max_label:", max_l, ",max_num:", max_num)

d_in = data_set.dim
d_h1 = 300
d_h2 = 100
d_out = data_set.dim
model = m.RereSVDD(d_in, d_h1, d_h2, d_out, max_l).to(cnf.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    hl.train(epoch, model, optimizer, train_loader,log_file)


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
    # 变换出DML数据
    trans_data = model.transform(data_set.data.to(cnf.device),labels).detach().cpu().numpy()
    # 可视化DML数据
    viz2 = tsne.RereTSNE(trans_data, labels)
    viz2.save_image(img3)
    # 保存DML变换后的数据
    hl.save_numpy_data_to_csv(trans_data, labels, output2)
    # 保存训练好的模型参数
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    hl.before_train(log_file)
    for epoch in range(1, epoch_num + 1):
        hl.train(epoch, model, optimizer, train_loader,log_file)
        # test(epoch)
    transform_save()
