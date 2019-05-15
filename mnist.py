import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import cv2


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imshow2(img):
    npimg = img
    plt.imshow(npimg)
    plt.show()

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))  
        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16*5*5)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)
        
        return x

def train(data, model):
    model.train()

    learning_rate = 0.1
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 4
    p = 1
    with open("loss.txt", "wt") as f:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, sample in enumerate(data, 0):
                optimizer.zero_grad()            
                
                inputs = sample[0]
                labels = sample[1]

                output = model(inputs)
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 500 == 499:    # print every 500 mini-batches
                    print('[%d, %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                    s = "{0} {1}\n".format(p, running_loss / 500)
                    f.write(s)
                    p += 1
                    running_loss = 0.0

    torch.save(model, './model.pth')


def validation(data, model):
    model.eval()
    print("Validating...")
    show_image = False  

    size = len(data)
    num_incorrect = 0
    i = 0
    for sample in data:
        images, labels = sample
        #img = transforms.functional.to_pil_image(images[0][0], mode='L')
        #img.save("img_{}.png".format(i), "png")
        output = model(images)
        predicted = torch.max(output.data, 1)
        if labels[0] != predicted[1].item():
            num_incorrect += 1
            if show_image: 
                s = "Real: {0}\t Predicted: {1}".format(labels[0], predicted[1].item())
                print(s)
                imshow(torchvision.utils.make_grid(images))
        i += 1
    print("Validation Error: {0} %".format(100.0 * num_incorrect / size))

def validationSlidingImages(images, model,x,y):
    #img = transforms.functional.to_pil_image(images[0][0], mode='L')
    #img.save("img_{}.png".format(i), "png")
    output = model(images)
    predicted = torch.max(output.data, 1)
    s = "Predicted value: {0}\t Class: {1}\t on position ({2},{3})".format(predicted[0].item(),predicted[1].item(),y,x)
    print(s)
    return predicted[0].item(),predicted[1].item()
    #imshow(torchvision.utils.make_grid(images))



def sliding_window(model, image, size):
    slide_offset = 5
    found_objects = []
    for i in range(0, image.shape[0]-(size[0]+slide_offset),slide_offset):
        for j in range(0,image.shape[1]-(size[1]+slide_offset),slide_offset):
            out_img = image[i:i+size[0],j:j+size[1]]
            img = np.reshape(out_img, (1, 1, size[0], size[1])) / 255
            img = torch.from_numpy(img)
            img = img.type(torch.FloatTensor)
            ''' SHOW IMAGE WITH PLT '''
            #imshow2(out_img)
            pred_value, pred_class = validationSlidingImages(img,model,i+size[0]/2,j+size[0]/2)
            if (pred_value > 12):
                found_objects.append((j,i,pred_class))
            

    ''' SHOW IMAGE WITH OPENCV '''

    # add roi's to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    for item in found_objects:
        cv2.putText(image,"{0}".format(item[2]),(item[0]+size[0]+3,item[1]+5), font, .3,(255,0,0),1,cv2.LINE_AA)
        cv2.rectangle(image,(item[0],item[1]),(item[0]+size[0],item[1]+size[1]),(255,0,0),1)
    cv2.imshow("image", image)
    cv2.waitKey(0)


def main():
    #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    #batch_size_train = 12

    #trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=batch_size_train, shuffle=True)
    #testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform))
    
    #model = LeNet()
    model = torch.load("./model.pth")
    #assert model != None, "model == null" 
    #for i in range(5):
    #train(trainloader, model)
    #validation(testloader, model)

    img = cv2.imread('./data/mnist/numbers.png', 0)
    sliding_window(model, img, (28, 28))

    
if __name__ == '__main__':
    main()  
