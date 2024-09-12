#Date 09-12-2024
#Author- Priyam Patel
# This is the model code for WAA with Variance term
import torchvision
import torch.nn as nn
class MobnetV3_WAA_variance(nn.Module):
        
        def __init__(self):
            super(MobnetV3_WAA_variance, self).__init__()
            #Mobilenet feature extractor
            self.model= torchvision.models.mobilenet_v3_large(weights=None)
            self.model = nn.Sequential(*(list(self.model.children())[:-1]))

            # Weight of a particular frame
            self.W=nn.Sequential(nn.Linear(960,64),nn.Tanh(),nn.Linear(64,1),nn.Sigmoid())

            # Reducing the feature space
            self.Q=nn.Linear(960,128)

            # Final fully connected layer
            self.P=nn.Linear(128,1) # GA branch
            self.variance=nn.Sequential(nn.Linear(128,1),nn.Softplus()) #Variance branch
            
        def forward(self,y,frames):
            #y is of dim [n_frames,3,256,256]
            #frames could be n_frames, but is an argument to send in multiple sweeps at once through the Feature Extractor and then (frames=n_frames per sweep)
            out1=self.model(y)

            out1reshaped=torch.reshape(out1,(int(out1.shape[0]/frames),frames,960))
            out2=self.W(out1reshaped)

            out2=out2/torch.sum(out2,dim=1,keepdim=True)

            out3=self.Q(out1reshaped)

            out4=self.P(torch.sum(out2*out3,dim=1,keepdim=True))

            variance=1e-6+ self.variance(torch.sum(out2*out3,dim=1,keepdim=True))
          
            return out4, variance
