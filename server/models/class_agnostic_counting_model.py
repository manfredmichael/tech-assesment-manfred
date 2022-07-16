"""
Basic class agnostic counting model with backbone, refiner, matcher and counter.
"""
import torch
from torch import nn
import math
import pytorch_lightning as pl

class CACModel(pl.LightningModule):
    """ Class Agnostic Counting Model"""
    
    def __init__(self, backbone, EPF_extractor, refiner, matcher, counter, hidden_dim, lr, decay, max_norm, criterion, device):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            EPF_extractor: torch module of the feature extractor for patches. See epf_extractor.py
            repeat_times: Times to repeat each exemplar in the transformer decoder, i.e., the features of exemplar patches.
        """
        super().__init__()
        self.EPF_extractor = EPF_extractor
        self.refiner = refiner
        self.matcher = matcher
        self.counter = counter

        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.lr = lr
        self.decay = decay
        self.device_ = device
        self.max_norm = max_norm
        self.criterion = criterion
        
    def forward(self, samples: torch.Tensor, patches: torch.Tensor, is_train: bool):
        """ The forward expects samples containing query images and corresponding exemplar patches.
            samples is a stack of query images, of shape [batch_size X 3 X H X W]
            patches is a torch Tensor, of shape [batch_size x num_patches x 3 x 128 x 128]
            The size of patches are small than samples

            It returns a dict with the following elements:
               - "density_map": Shape= [batch_size x 1 X h_query X w_query]
               - "patch_feature": Features vectors for exemplars, not available during testing.
                                  They are used to compute similarity loss. 
                                Shape= [exemplar_number x bs X hidden_dim]
               - "img_feature": Feature maps for query images, not available during testing.
                                Shape= [batch_size x hidden_dim X h_query X w_query]
            
        """
        # Stage 1: extract features for query images and exemplars
        scale_embedding, patches = patches['scale_embedding'], patches['patches']
        features = self.backbone(samples)
        features = self.input_proj(features)
        
        patches = patches.flatten(0, 1) 
        patch_feature = self.backbone(patches) # obtain feature maps for exemplar patches
        patch_feature = self.EPF_extractor(patch_feature, scale_embedding) # compress the feature maps into vectors and inject scale embeddings
        
        # Stage 2: enhance feature representation, e.g., the self similarity module.
        refined_feature, patch_feature = self.refiner(features, patch_feature)
        # Stage 3: generate similarity map by densely measuring similarity. 
        counting_feature, corr_map = self.matcher(refined_feature, patch_feature)
        # Stage 4: predicting density map 
        density_map = self.counter(counting_feature)
        
        if not is_train:
            return density_map
        else:
            return {'corr_map': corr_map, 'density_map': density_map}

    #def _reset_parameters(self):
    #    for p in self.parameters():
    #        if p.dim() > 1:
    #            nn.init.xavier_uniform_(p)

    def training_step(self, batch, batch_nb):
        img, patches, targets = batch 
        img = img.to(self.device_)
        patches['patches'] = patches['patches'].to(self.device_)
        patches['scale_embedding'] = patches['scale_embedding'].to(self.device_)
        density_map = targets['density_map'].to(self.device_)
        pt_map = targets['pt_map'].to(self.device_)

        outputs = self(img, patches, is_train=True)

        dest = outputs['density_map']
        if batch_nb < 5: # check if training process get stucked in local optimal. 
            print(dest.sum().item(), density_map.sum().item(), dest.sum().item()*10000 / (img.shape[-2] * img.shape[-1]))
        counting_loss, contrast_loss = self.criterion(outputs, density_map, pt_map)
        loss = counting_loss if isinstance(contrast_loss, int) else counting_loss + contrast_loss
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            return None
            

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)

        self.log("train_loss", loss, on_epoch=True)

        return loss



    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.decay)
