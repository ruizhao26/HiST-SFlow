import torch
import torch.nn as nn

class Rep3DConv(nn.Module):
    def __init__(self):
        super(Rep3DConv, self).__init__()
        
        self.T = 25

        # Res 1/1 -> Res 1/2: Conv3d
        self.conv3d_to_res_1_2 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5,3,3), padding=(0,1,1), stride=(1,2,2)),  # T: (25 - 5 + 1) = 21 
            # nn.InstanceNorm3d(16),
            nn.ReLU(),
        )

        # Res 1/2 -> Res 1/4: Conv3d
        self.conv3d_to_res_1_4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5,3,3), padding=(0,1,1), stride=(2,2,2)),  # T: ((21 - 5 + 1) + 1) / 2 = 9
            # nn.InstanceNorm3d(24),
            nn.ReLU(),
        )

        # Res 1/4 -> Res 1/8: Conv3d
        self.conv3d_to_res_1_8 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=(5,3,3), padding=(0,1,1), stride=(2,2,2)),  # T: ((9 - 5 + 1) + 1) / 2 = 3
            # nn.InstanceNorm3d(32),
            nn.ReLU(),
        )

        
        # Res 1/2 Enc and Dec
        self.conv_intra_1_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(16),
            nn.ReLU(),
        )
        self.conv_fuse_1_2 = nn.Conv2d(16*21, 32, kernel_size=3, padding=1)
        self.deconv_res1_2 = nn.ConvTranspose2d(32+64, 64, kernel_size=4, padding=1, stride=2)
        
        # Res 1/4 Enc and Dec
        self.conv_intra_1_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(24),
            nn.ReLU(),
        )
        self.conv_fuse_1_4 = nn.Conv2d(32*9, 48, kernel_size=3, padding=1)
        self.deconv_res1_4 = nn.ConvTranspose2d(48+64, 64, kernel_size=4, padding=1, stride=2)
        
        # Res 1/4 Enc and Dec
        self.conv_intra_1_8 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.conv_fuse_1_8 = nn.Conv2d(48*3, 64, kernel_size=3, padding=1)
        self.deconv_res1_8 = nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2)

        # output
        self.out_layer = nn.Conv2d(64, 128, kernel_size=3, padding=1)


    def trans_3d_to_2d_separate(self, x):
        # B C T H W -> B T C H W -> (B*T) C H W
        B, C, self.T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B*self.T, C, H, W)
        return x
    
    def trans_2d_to_2d_for_fuse(self, x):
        # (B*T) C H W -> B T C H W -> B (T*C) H W
        BT, C, H, W = x.shape
        B = BT // self.T
        x = x.view(B, self.T, C, H, W).view(B, self.T*C, H, W)
        return x
    
    def trans_2d_separate_to_3d(self, x):
        # (B*T) C H W -> B C T H W
        BT, C, H, W = x.shape
        B = BT // self.T
        x = x.view(B, self.T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return x


    def forward(self, x):
        # # feature for concat: Res 1/1
        # cat_feat1 = self.conv_res1_1a(x)
        # short_cut = cat_feat1
        # cat_feat1 = short_cut + self.conv_res1_1b(cat_feat1)

        ########################################################################
        # Conv 3D downsample: Res 1/1 -> 1/2
        f3d = x.unsqueeze(dim=1)            # B T H W -> B 1 T H W
        f3d = self.conv3d_to_res_1_2(f3d)

        # feature for concat: Res 1/2
        f2d = self.trans_3d_to_2d_separate(f3d)
        intra_feat2 = f2d + self.conv_intra_1_2(f2d)    # residual
        feat2_for_fuse = self.trans_2d_to_2d_for_fuse(intra_feat2)
        cat_feat2 = self.conv_fuse_1_2(feat2_for_fuse)

        ########################################################################
        # Conv 3D downsample: Res 1/2 -> 1/4
        f3d = self.conv3d_to_res_1_4(self.trans_2d_separate_to_3d(intra_feat2))

        # feature for concat: Res 1/4
        f2d = self.trans_3d_to_2d_separate(f3d)
        intra_feat4 = f2d + self.conv_intra_1_4(f2d)    # residual
        feat4_for_fuse = self.trans_2d_to_2d_for_fuse(intra_feat4)
        cat_feat4 = self.conv_fuse_1_4(feat4_for_fuse)

        ########################################################################
        # Conv 3D downsample: Res 1/2 -> 1/4
        f3d = self.conv3d_to_res_1_8(self.trans_2d_separate_to_3d(intra_feat4))

        # feature for concat: Res 1/4
        f2d = self.trans_3d_to_2d_separate(f3d)
        intra_feat8 = f2d + self.conv_intra_1_8(f2d)    # residual
        feat8_for_fuse = self.trans_2d_to_2d_for_fuse(intra_feat8)
        cat_feat8 = self.conv_fuse_1_8(feat8_for_fuse)

        feat8 = self.deconv_res1_8(cat_feat8)

        feat4 = self.deconv_res1_4(torch.cat([feat8, cat_feat4], dim=1))
        feat2 = self.deconv_res1_2(torch.cat([feat4, cat_feat2], dim=1))

        out = self.out_layer(feat2)

        return [out, cat_feat2, cat_feat4, cat_feat8]
