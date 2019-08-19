# -*- coding: utf-8 -*-
"""
## CIFAR-10 画像分類  
飛行機、自動車、鳥、猫など (合計10個カテゴリ

それぞれの訓練用5000枚、テスト用1000枚   
合計 6000枚 x 10 = 60000枚のデータセット  

画像 32 x 32 ピクセル  
RGBの3チャンネル (3, 32, 32)
"""

# モジュールの読み込み
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# %matplotlib inline
import numpy as np
from matplotlib import pyplot as plt

"""### データ処理の定義"""

# データ拡張あり(訓練データ)
# 訓練データをLoadする場合に、データの処理を記述する。以下の処理はデータ拡張を行う処理である。
transform_train= transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# データ拡張なし(訓練データ)
transform_train_non_aug= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# テストデータ
# 訓練データと同様に正規化を行う。ただしデータ拡張は実施しない。
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

"""### データセットのダウンロードとロード  
学習に用いるデータについて画像拡張のあり/なしを以下のコードにて洗濯する。
"""

# 画像拡張あり
train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=transform_train, download=True )

# 画像拡張なし(ありを選ぶ場合には、以下の行をコメントアウトする)
# train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=transform_train_non_aug, download=True )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=transform_test, download=True )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

!ls ../data/cifar-10-batches-py/

"""ダウンロード後回答したファイルは以下のようになっている。

batches.meta - ラベル名のリスト  
data_batch_1 - 訓練画像（1万枚）  
data_batch_2 - 訓練画像（1万枚）  
data_batch_3 - 訓練画像（1万枚）  
data_batch_4 - 訓練画像（1万枚）  
data_batch_5 - 訓練画像（1万枚）  
test_batch - テスト画像（1万枚） 

訓練画像は1万枚ずつ別のファイルになっている
"""

# ダウンロードしたデータの確認（訓練データ）
image, label = train_dataset[0]

# imageのサイズ
print("imageのサイズ", image.size())

print()

# 訓練データの長さ
print('訓練データの長さ:', len(train_dataset[0]))

print("訓練データの表示:タプル（データ, 教師ラベル）")
# 訓練データの表示　タプル（データ, 教師ラベル）
print("訓練データの表示", train_dataset[0])

print()

# imageの表示
print("image", image)

# labelの表示
print("label", label)

"""(確認した結果)  
train_dataset[0]は、要素を２つもったタプル  
(画像データ([3,32,32]), ラベル)
"""

# ダウンロードしたデータの確認（テストデータ）
image, label = test_dataset[0]
print(test_dataset[0])
print('len:', len(test_dataset[0]))
print(image)
print(image.size())
print(label)

"""(確認した結果)  
test_dataset[0]は、要素を２つもったタプル  
(画像データ([3,32,32]), ラベル)
"""

print('train_dataset:', len(train_dataset))
print('test_dataset:', len(test_dataset))

"""### 画像の表示"""

# 表示用にデータをロードする
img_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
classes_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 画像の表示
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return  

# 訓練用のデータをランダムに得る
dataiter = iter(img_loader)
images, labels = dataiter.next()
 
# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes_labels[labels[j]] for j in range(5)))

"""## モデルの作成  
nn.Conv2d
入力：形状（バッチサイズ、チャンネル数、縦、横）
出力：形状（バッチサイズ、チャンネル数、縦、横）
"""

# 分類するクラス数(CIFAR-10は10クラスに分類される画像セットであるため)
num_classes = 10

"""torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros')"""

class MyNet(nn.Module):

    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        
        # 特徴量抽出(畳み込み抽出) 
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(4096, num_classes)

    # 全結合    
    def forward(self, x):
        # 特徴量抽出
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

"""## 損失関数と最適化アルゴリズムの定義"""

# deviceの選択
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('torch.cuda.is_available():', torch.cuda.is_available())
print('device:', device)

net = MyNet(num_classes).to(device)

# 損失関数の定義（クロスエントロピーを使用する）
criterion = nn.CrossEntropyLoss()

# モデルパラメータの更新手法
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

"""ネットワークを構築後、デバイスの選択、損失関数の定義、モデルパラメータ更新手法を定義した"""

# 定義したモデルを表示する
net

num_epochs = 20

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    
    #train（訓練）
    net.train()
    for i, (images, labels) in enumerate(train_loader):
      #view()での変換をしない
      images, labels = images.to(device), labels.to(device)
      
      # 前回backwardメソッドで計算された勾配の値を削除
      optimizer.zero_grad()
      # モデルの出力
      outputs = net(images)
      # 損失関数 (出力とラベル)
      loss = criterion(outputs, labels)
      # 損失
      train_loss += loss.item()
      # 精度の確認
      train_acc += (outputs.max(1)[1] == labels).sum().item()
      
      #if epoch == 0:
      #    print('outputs.max(1):', outputs.max(1))
      #    print('outputs.max(1)[0]:', outputs.max(1)[0])
      #    print('outputs.max(1)[1]:',outputs.max(1)[1])
      #    print('label',labels)
      
      loss.backward()
      optimizer.step()
    
    # 平均（損失と精度）
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)
    
    
    #validation (テストデータでの確認)
    net.eval()
    with torch.no_grad():
      for images, labels in test_loader:
        #view()での変換をしない
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_acc += (outputs.max(1)[1] == labels).sum().item()
    
    # 平均（損失と精度）
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    
    # 平均の訓練データの損失(loss)とテストデータの損失(val_loss)、テストデータの分類精度
    print ('Epoch [{}/{}], Loss: {loss:.4f}, train_acc:{train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, train_acc=avg_train_acc, val_loss=avg_val_loss, val_acc=avg_val_acc))
    
    # 各Epochのそれぞれの値をリストに追加（学習の履歴記録、グラフ作成用）
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

# グラフの描画
import matplotlib.pyplot as plt
# %matplotlib inline

# 学習時の損失
plt.figure()
# 訓練データを使った損失をプロット（青の実線）
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
# 訓練データを使った損失をプロット（緑の点線）
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()

# 学習時の精度
plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()