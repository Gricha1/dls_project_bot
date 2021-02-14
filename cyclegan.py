#Загружаем требуемые библиотеки для загрузки данных и транформеры
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
import warnings

# %matplotlib inline

#Связываем нотбук с гугл диском
from google.colab import drive
drive.mount('/content/drive')

#Суть функции - получить генераторы батчей изображений(сжатых до размеров image_size) с заранее подготовленной
#папки с гугл диска
def get_data_loader(image_type, image_dir='/content/drive/MyDrive/summer2winter_yosemite', 
                    image_size=128, batch_size=16, num_workers=0):
    
    
    transform = transforms.Compose([transforms.Resize(image_size), 
                                    transforms.ToTensor()])
    

    #заранее созданная папка с изображениями имеет путь - image dir
    image_path =  image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    #получение самих генераторов 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

dataloader_X, test_dataloader_X = get_data_loader(image_type='summer')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter')

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#Наши изображения являются тензорами с диапазоном значений (0, 1)
#функция scale изменяет диапазон картинки на (-1, +1) потому что так лучше работает))  
def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    x = x * (max - min) + min
    return x

import torch.nn as nn
import torch.nn.functional as F

#функция conv - возвращает последовательность слоев нейросети - Conv2d и Batchnorm(если указан) 
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

#определяем наш дискриминатор - бинарный классификатор на фейк-рил изображение
class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        self.conv1 = conv(3,64,4,batch_norm=False)
        self.conv2 = conv(64,128,4)
        self.conv3 = conv(128,256,4)
        self.conv4 = conv(256,512,4)
        
        self.conv5 = conv(512,1,4,1,batch_norm=False)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        
        return x

# Bottleneck будущего генератора(часть между енкодером и декодером) задается с помощью класса ResidualBlock 
class ResidualBlock(nn.Module):
    #conv_dim - колличество входных карт активаций в residualblock 
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(conv_dim,conv_dim,3,1)
        self.conv2 = conv(conv_dim,conv_dim,3,1)
        
    def forward(self, x):
        input_x = x
        x = F.relu(self.conv1(x))
        x = input_x + self.conv2(x)
        return x

#Так же как с енкодером прописываем вспомогательную функцию deconv для определения
#слоев сети в части декодера
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

#Определение самого генератора вид - енкодер, bottleneck, декодер
class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        #Encoder
        self.enc_conv1 = conv(3,conv_dim,4,2)
        self.enc_conv2 = conv(conv_dim,conv_dim*2,4,2)
        self.enc_conv3 = conv(conv_dim*2,conv_dim*4,4,2)
        

        #BottleNeck
        l = [ResidualBlock(conv_dim*4) for i in range(n_res_blocks)]
        self.resBlock = nn.Sequential(*l)

        #Decoder
        self.dec_conv1 = deconv(conv_dim*4,conv_dim*2,4,2)
        self.dec_conv2 = deconv(conv_dim*2,conv_dim,4,2)
        self.dec_conv3 = deconv(conv_dim,3,4,2)
       

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        
        x = self.resBlock(x)
        
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.tanh(self.dec_conv3(x))
        return x

#функция create_model - создает конечную модель состоящую из двух генераторов 
#Из множества картинок с летом во множество картинок с зимой и наоборот
#А так же из двух соотвествующих им декодеров
def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    
    
    #Генераторы
    G_XtoY = CycleGenerator(g_conv_dim,n_res_blocks)
    G_YtoX = CycleGenerator(g_conv_dim,n_res_blocks)
    #Дискриминаторы
    D_X = Discriminator(d_conv_dim)
    D_Y = Discriminator(d_conv_dim)
    #Кидаем все что можем на GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y

G_XtoY, G_YtoX, D_X, D_Y = create_model()

#Лосс функция потребуется для обучения дискриминатора - будет сравнивать его выход с реальным изображанием
def real_mse_loss(D_out):
    return torch.mean((D_out-1)**2)
#Лосс функция, которая высчитывает лосс предсказания дискриминатора на фековых изображениях
def fake_mse_loss(D_out):
    return torch.mean(D_out**2)

#Лосс функция предназаченая для оценки реконструкции
def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    return torch.mean(torch.abs(real_im-reconstructed_im)) * lambda_weight

#В данном блоке задаем оптимизатор для нашей модели(Адам) и определяем гиперпараметры
import torch.optim as optim
lr= 0.0002
beta1= 0.5
beta2= 0.999

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters()) 

g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

#Функция обучения нашей модели 
def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, 
                  n_epochs=1000):
    

    print_every = 10
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)



    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs+1):

        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X) 

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)
        
  
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)

    #Тренеровка ДИСКРИМИНАТОРА (D_X, D_Y)
        d_x_optimizer.zero_grad()
        #1. Имея два типа лосса вначале мы высчитываем ошибку на реальном изображении лета
        #далее мы генерируем изображение лета и высчитываем уже ошибку фейка
        D_X_real_loss = real_mse_loss(D_X(images_X))
        G_Y2X_fake_image = G_YtoX(images_Y) 
        D_X_fake_loss = fake_mse_loss(D_X(G_Y2X_fake_image))
        
        #суммарная ошибка и бэк проп у первого дискриминатора
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()
        
        #2. делаем аналогичные действия (пункт 1.) для второго генератора
        D_Y_real_loss = real_mse_loss(D_Y(images_Y))
        G_X2Y_fake_image = G_XtoY(images_X) 
        D_Y_fake_loss = fake_mse_loss(D_Y(G_X2Y_fake_image))
        
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        
        d_y_loss.backward()
        d_y_optimizer.step()


    #Тренеровка ГЕНЕРАТОРА(G_YtoX, G_XtoY)
        g_optimizer.zero_grad()

        #1. Вначале первым генератором генерируем изображение лета G_X_img 
        #далее наш дискриминатор на сгенерированном изображении позвоялет оценить насколько мы ошиблись
        #по сгенерированному изображению G_X_img восстанавливаем изображение зимы вторым генератором
        G_X_img = G_YtoX(images_Y)
        G_X_real_loss = real_mse_loss(D_X(G_X_img))
        G_Y_reconstructed = G_XtoY(G_X_img)
        #Высчитываем последнюю ошибку на полученном изображении зимы 
        G_Y_consistency_loss = cycle_consistency_loss(images_Y,G_Y_reconstructed,10)


        #2. Повторяем те же самые действия(что в пункте 1), но с первоначальной генерацией изображения зимы
        G_Y_img = G_XtoY(images_X)
        G_Y_real_loss = real_mse_loss(D_Y(G_Y_img))
        G_X_reconstructed = G_YtoX(G_Y_img)
        G_X_consistency_loss = cycle_consistency_loss(images_X,G_X_reconstructed,10)
    
        #Суммарная ошибка, бэкпроп и все дела
        g_total_loss = G_X_real_loss + G_Y_real_loss + G_Y_consistency_loss + G_X_consistency_loss
        g_total_loss.backward()
        g_optimizer.step()
        
      
      #Вывод ошибки при обучении
        if epoch % print_every == 0:
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
        
    return losses

#Запускаем обучение, достаточное колличество эпох - 3000
n_epochs = 4000

losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_epochs=n_epochs)

torch.save(G_XtoY.state_dict(), 'drive/MyDrive/weights_sum_wint.pth')
torch.save(G_YtoX.state_dict(), 'drive/MyDrive/weights_wint_sum.pth')
