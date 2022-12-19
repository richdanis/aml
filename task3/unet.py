
import torch
import torch.nn as nn

class Unet(nn.Module):
    
    def __init__(self, filters=64, kernel=3):
        
        super().__init__()
        
        self.kernel = kernel

        self.max_pool = nn.MaxPool2d(2)
        
        self.block_enc_1 = self.conv_block(1, filters)
        self.block_enc_2 = self.conv_block(filters, 2*filters)
        self.block_enc_3 = self.conv_block(2*filters, 4*filters)
        self.block_enc_4 = self.conv_block(4*filters, 8*filters)
        
        self.block_inbetween = self.conv_block(8*filters, 16*filters, True)
        
        self.block_dec_1 = self.conv_block(16*filters, 8*filters, True)
        self.block_dec_2 = self.conv_block(8*filters, 4*filters, True)
        self.block_dec_3 = self.conv_block(4*filters, 2*filters, True)
        
        self.block_last = self.conv_block(2*filters, filters, True, True)
        
    def conv_block(self, channels, filters, dec=False, last=False):
        
        modules = []
        
        if not dec:
            modules.append(nn.Dropout(p=0.5))
            
        # modules.append(torch.nn.BatchNorm2d(channels))
        
        modules.append(nn.Conv2d(channels, filters, self.kernel, 1, padding='same'))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(filters, filters, self.kernel, 1, padding='same'))
        modules.append(nn.ReLU())
        
        # if not decoder, then we add upscaling layer
        # if last layer, then we add conv 1x1 and sigmoid to get logits in (0, 1)
        if dec:
            if not last:
                modules.append(nn.ConvTranspose2d(filters, filters//2, 2, stride=(2,2)))
            else:
                modules.append(nn.Conv2d(filters, 1, 1, 1))
                modules.append(nn.Sigmoid())
            
        return nn.Sequential(*modules)
            
    def forward(self, x):
        
        # encoder
        
        x1 = self.block_enc_1(x)
        x2 = self.max_pool(x1)
        
        x3 = self.block_enc_2(x2)
        x4 = self.max_pool(x3)
        
        x5 = self.block_enc_3(x4)
        x6 = self.max_pool(x5)

        x7 = self.block_enc_4(x6)
        x8 = self.max_pool(x7)
        
        # between encoder and decoder
        
        x9 = self.block_inbetween(x8)
        
        # decoder
        
        x10 = self.block_dec_1(torch.cat((x9, x7), dim=1))
        x11 = self.block_dec_2(torch.cat((x10, x5), dim=1))
        x12 = self.block_dec_3(torch.cat((x11, x3), dim=1))

        x13 = self.block_last(torch.cat((x12, x1), dim=1))
        
        return x13