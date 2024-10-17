import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .base import BasePredictor
from isegm.model.is_hrnet_model import DistMapsHRNetModel


class BRSBasePredictor(BasePredictor):
    def __init__(self, model, device, opt_functor, optimize_after_n_clicks=1, **kwargs):
        super().__init__(model, device, **kwargs)
        self.optimize_after_n_clicks = optimize_after_n_clicks
        self.opt_functor = opt_functor

        self.opt_data = None
        self.input_data = None

    def set_input_image(self, image_nd):
        super().set_input_image(image_nd)
        self.opt_data = None
        self.input_data = None

    def _get_clicks_maps_nd(self, clicks_lists, image_shape, radius=1):
        pos_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        neg_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)

        for list_indx, clicks_list in enumerate(clicks_lists):
            for click in clicks_list:
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                y1, x1 = y - radius, x - radius
                y2, x2 = y + radius + 1, x + radius + 1

                if click.is_positive:
                    pos_clicks_map[list_indx, 0, y1:y2, x1:x2] = True
                else:
                    neg_clicks_map[list_indx, 0, y1:y2, x1:x2] = True

        with torch.no_grad():
            pos_clicks_map = torch.from_numpy(pos_clicks_map).to(self.device, dtype=torch.float)
            neg_clicks_map = torch.from_numpy(neg_clicks_map).to(self.device, dtype=torch.float)

        return pos_clicks_map, neg_clicks_map

    def get_states(self):
        return {'transform_states': self._get_transform_states(), 'opt_data': self.opt_data}

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.opt_data = states['opt_data']

    
    ##########################################################   

    def _get_newest_click_map(self, clicks_lists, image_shape, radius=8): #disk
        newest_click_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        
        click = clicks_lists[-1][-1]
        y, x = click.coords
        y, x = int(round(y)), int(round(x))
        
        radius = click.radius
        radius = int(round(radius))
            
        y1, x1 = y - radius, x - radius
        y2, x2 = y + radius + 1, x + radius + 1
        
        y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
        mask = x*x + y*y <= radius*radius
        newest_click_map[0, 0, mask] = True
        
        zeros = np.zeros_like(newest_click_map)
        if click.is_positive:
            click4agm = np.concatenate((zeros, newest_click_map), axis=1)
        else:
            click4agm = np.concatenate((newest_click_map, zeros), axis=1)
                
        with torch.no_grad():
            newest_click_map = torch.from_numpy(newest_click_map).to(self.device, dtype=torch.float)
            click4agm = torch.from_numpy(click4agm).to(self.device, dtype=torch.float)
            
        return newest_click_map, click4agm, radius, click.is_positive
    
    def _get_newest_click_mask(self, clicks_lists, image_shape, radius=8):
        newest_click_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        
        click = clicks_lists[-1][-1]
        y, x = click.coords
        y, x = int(round(y)), int(round(x))
        
        radius = click.radius * 10
        radius = int(round(radius))
            
        y1, x1 = y - radius, x - radius
        y2, x2 = y + radius + 1, x + radius + 1
        
        newest_click_map[0, 0, y1:y2, x1:x2] = True
                
        with torch.no_grad():
            newest_click_mask = torch.from_numpy(newest_click_map).to(self.device, dtype=torch.float)
            
        return newest_click_mask
    
    def _get_newest_click_map_vis(self, clicks_lists, image_shape, radius=8): #disk
        newest_click_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        
        click = clicks_lists[-1][-1]
        y, x = click.coords
        y, x = int(round(y)), int(round(x))
        
        radius = click.radius
        radius = int(round(radius))
            
        y1, x1 = y - radius, x - radius
        y2, x2 = y + radius + 1, x + radius + 1
        
        y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
        mask = x*x + y*y <= radius*radius
        newest_click_map[0, 0, mask] = True
        
        
        y, x = click.coords
        y, x = int(round(y)), int(round(x))
        
        radius = radius - 1
        
        y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
        mask = x*x + y*y <= radius*radius
        newest_click_map[0, 0, mask] = False
        
        y, x = click.coords
        y, x = int(round(y)), int(round(x))
        
        radius = 2
        
        y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
        mask = x*x + y*y <= radius*radius
        newest_click_map[0, 0, mask] = True
        
        zeros = np.zeros_like(newest_click_map)
        if click.is_positive:
            click4cgam = np.concatenate((zeros, newest_click_map), axis=1)
        else:
            click4cgam = np.concatenate((newest_click_map, zeros), axis=1)
                
        with torch.no_grad():
            newest_click_map = torch.from_numpy(newest_click_map).to(self.device, dtype=torch.float)
            # save_image(newest_click_map, f'./Disk_New_Click/disk_{time()}.png')
            click4cgam = torch.from_numpy(click4cgam).to(self.device, dtype=torch.float)
            
        return newest_click_map, click4cgam, radius, click.is_positive
    
    
    def _get_CGAM_click_maps(self, clicks_lists, image_shape, radius=1):
        pos_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        neg_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)

        for list_indx, clicks_list in enumerate(clicks_lists):
            for click in clicks_list:
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                
                radius = click.radius
                radius = int(round(radius))
                
                y1, x1 = y - radius, x - radius
                y2, x2 = y + radius + 1, x + radius + 1
                
                y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
                mask = x*x + y*y <= radius*radius

                if click.is_positive:
                    pos_clicks_map[0, 0, mask] = True
                else:
                    neg_clicks_map[0, 0, mask] = True

        with torch.no_grad():
            pos_clicks_map = torch.from_numpy(pos_clicks_map).to(self.device, dtype=torch.float)
            neg_clicks_map = torch.from_numpy(neg_clicks_map).to(self.device, dtype=torch.float)
            click_maps = torch.cat((pos_clicks_map, neg_clicks_map), 1)
        return click_maps
    
    
    def _get_click_maps_vis(self, clicks_lists, image_shape, radius=1):
        pos_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        neg_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)

        for list_indx, clicks_list in enumerate(clicks_lists):
            for click in clicks_list:
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                
                radius = click.radius
                radius = int(round(radius))
                
                y1, x1 = y - radius, x - radius
                y2, x2 = y + radius + 1, x + radius + 1
                
                y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
                mask = x*x + y*y <= radius*radius

                if click.is_positive:
                    pos_clicks_map[0, 0, mask] = True
                else:
                    neg_clicks_map[0, 0, mask] = True
                    
                
                radius = radius - 2
                
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
                mask = x*x + y*y <= radius*radius

                if click.is_positive:
                    pos_clicks_map[0, 0, mask] = False
                else:
                    neg_clicks_map[0, 0, mask] = False
                    
                radius = 5
                
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
                mask = x*x + y*y <= radius*radius

                if click.is_positive:
                    pos_clicks_map[0, 0, mask] = True
                else:
                    neg_clicks_map[0, 0, mask] = True
                

        with torch.no_grad():
            pos_clicks_map = torch.from_numpy(pos_clicks_map).to(self.device, dtype=torch.float)
            neg_clicks_map = torch.from_numpy(neg_clicks_map).to(self.device, dtype=torch.float)
            click_maps = torch.cat((pos_clicks_map, neg_clicks_map), 1)
        return click_maps
    
    
    def _get_click_maps_vis_fBRS(self, clicks_lists, image_shape, radius=5):
        pos_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        neg_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)

        for list_indx, clicks_list in enumerate(clicks_lists):
            for click in clicks_list:
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                
                y, x = np.ogrid[-y:image_shape[0]-y, -x:image_shape[1]-x]
                mask = x*x + y*y <= radius*radius

                if click.is_positive:
                    pos_clicks_map[0, 0, mask] = True
                else:
                    neg_clicks_map[0, 0, mask] = True

        with torch.no_grad():
            pos_clicks_map = torch.from_numpy(pos_clicks_map).to(self.device, dtype=torch.float)
            neg_clicks_map = torch.from_numpy(neg_clicks_map).to(self.device, dtype=torch.float)
            click_maps = torch.cat((pos_clicks_map, neg_clicks_map), 1)
        return click_maps
    
    def _get_random_click_maps(self, clicks_lists, image_shape, radius=1):
        pos_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        neg_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)

        for list_indx, clicks_list in enumerate(clicks_lists):
            for click in clicks_list:
                y = np.random.randint(low=15, high=image_shape[0]-17, size=1)[0]
                x = np.random.randint(low=15, high=image_shape[1]-17, size=1)[0]
                
                radius = 15
                
                y1, x1 = y - radius, x - radius
                y2, x2 = y + radius + 1, x + radius + 1
                
                k = random.randint(0, 1)

                if k == 0:
                    pos_clicks_map[list_indx, 0, y1:y2, x1:x2] = True
                elif k ==1 :
                    neg_clicks_map[list_indx, 0, y1:y2, x1:x2] = True
                else:
                    print('error')

        with torch.no_grad():
            pos_clicks_map = torch.from_numpy(pos_clicks_map).to(self.device, dtype=torch.float)
            neg_clicks_map = torch.from_numpy(neg_clicks_map).to(self.device, dtype=torch.float)
            click_maps = torch.cat((pos_clicks_map, neg_clicks_map), 1)
        return click_maps


class FeatureBRSPredictor(BRSBasePredictor):
    def __init__(self, model, device, opt_functor, insertion_mode='after_deeplab', **kwargs):
        super().__init__(model, device, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
        self._c1_features = None

        if self.insertion_mode == 'after_deeplab':
            self.num_channels = model.feature_extractor.ch
        elif self.insertion_mode == 'after_c4':
            self.num_channels = model.feature_extractor.aspp_in_channels
        elif self.insertion_mode == 'after_aspp':
            self.num_channels = model.feature_extractor.ch + 32
        else:
            raise NotImplementedError

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])

        num_clicks = len(clicks_lists[0])
        bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]

        if self.opt_data is None or self.opt_data.shape[0] // (2 * self.num_channels) != bs:
            self.opt_data = np.zeros((bs * 2 * self.num_channels), dtype=np.float32)

        if num_clicks <= self.net_clicks_limit or is_image_changed or self.input_data is None:
            self.input_data = self._get_head_input(image_nd, points_nd)

        def get_prediction_logits(scale, bias):
            scale = scale.view(bs, -1, 1, 1)
            bias = bias.view(bs, -1, 1, 1)
            if self.with_flip:
                scale = scale.repeat(2, 1, 1, 1)
                bias = bias.repeat(2, 1, 1, 1)

            scaled_backbone_features = self.input_data * scale
            scaled_backbone_features = scaled_backbone_features + bias
            if self.insertion_mode == 'after_c4':
                x = self.net.feature_extractor.aspp(scaled_backbone_features)
                x = F.interpolate(x, mode='bilinear', size=self._c1_features.size()[2:],
                                  align_corners=True)
                x = torch.cat((x, self._c1_features), dim=1)
                scaled_backbone_features = self.net.feature_extractor.head(x)
            elif self.insertion_mode == 'after_aspp':
                scaled_backbone_features = self.net.feature_extractor.head(scaled_backbone_features)

            pred_logits = self.net.head(scaled_backbone_features)
            pred_logits = F.interpolate(pred_logits, size=image_nd.size()[2:], mode='bilinear',
                                        align_corners=True)
            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask, self.device)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data,
                                       **self.opt_functor.optimizer_params)
            self.opt_data = opt_result[0]

        with torch.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_data_nd = torch.from_numpy(self.opt_data).to(self.device)
                opt_vars, _ = self.opt_functor.unpack_opt_params(opt_data_nd)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits

    def _get_head_input(self, image_nd, points):
        with torch.no_grad():
            coord_features = self.net.dist_maps(image_nd, points)
            x = self.net.rgb_conv(torch.cat((image_nd, coord_features), dim=1))
            if self.insertion_mode == 'after_c4' or self.insertion_mode == 'after_aspp':
                c1, _, c3, c4 = self.net.feature_extractor.backbone(x)
                c1 = self.net.feature_extractor.skip_project(c1)

                if self.insertion_mode == 'after_aspp':
                    x = self.net.feature_extractor.aspp(c4)
                    x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=True)
                    x = torch.cat((x, c1), dim=1)
                    backbone_features = x
                else:
                    backbone_features = c4
                    self._c1_features = c1
            else:
                backbone_features = self.net.feature_extractor(x)[0]

        return backbone_features


class HRNetFeatureBRSPredictor(BRSBasePredictor):
    def __init__(self, model, device, opt_functor, insertion_mode='A', **kwargs):
        super().__init__(model, device, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
        self._c1_features = None

        if self.insertion_mode == 'A':
            self.num_channels = sum(k * model.feature_extractor.width for k in [1, 2, 4, 8])
        elif self.insertion_mode == 'C':
            self.num_channels = 2 * model.feature_extractor.ocr_width
        else:
            raise NotImplementedError

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])
        num_clicks = len(clicks_lists[0])
        bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]

        if self.opt_data is None or self.opt_data.shape[0] // (2 * self.num_channels) != bs:
            self.opt_data = np.zeros((bs * 2 * self.num_channels), dtype=np.float32)

        if num_clicks <= self.net_clicks_limit or is_image_changed or self.input_data is None:
            self.input_data = self._get_head_input(image_nd, points_nd)

        def get_prediction_logits(scale, bias):
            scale = scale.view(bs, -1, 1, 1)
            bias = bias.view(bs, -1, 1, 1)
            if self.with_flip:
                scale = scale.repeat(2, 1, 1, 1)
                bias = bias.repeat(2, 1, 1, 1)

            scaled_backbone_features = self.input_data * scale
            scaled_backbone_features = scaled_backbone_features + bias
            if self.insertion_mode == 'A':
                out_aux = self.net.feature_extractor.aux_head(scaled_backbone_features)
                feats = self.net.feature_extractor.conv3x3_ocr(scaled_backbone_features)

                context = self.net.feature_extractor.ocr_gather_head(feats, out_aux)
                feats = self.net.feature_extractor.ocr_distri_head(feats, context)
                pred_logits = self.net.feature_extractor.cls_head(feats)
            elif self.insertion_mode == 'C':
                pred_logits = self.net.feature_extractor.cls_head(scaled_backbone_features)
            else:
                raise NotImplementedError

            pred_logits = F.interpolate(pred_logits, size=image_nd.size()[2:], mode='bilinear',
                                        align_corners=True)
            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask, self.device)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data,
                                       **self.opt_functor.optimizer_params)
            self.opt_data = opt_result[0]

        with torch.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_data_nd = torch.from_numpy(self.opt_data).to(self.device)
                opt_vars, _ = self.opt_functor.unpack_opt_params(opt_data_nd)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits

    def _get_head_input(self, image_nd, points):
        with torch.no_grad():
            coord_features = self.net.dist_maps(image_nd, points)
            x = self.net.rgb_conv(torch.cat((image_nd, coord_features), dim=1))
            feats = self.net.feature_extractor.compute_hrnet_feats(x)
            if self.insertion_mode == 'A':
                backbone_features = feats
            elif self.insertion_mode == 'C':
                out_aux = self.net.feature_extractor.aux_head(feats)
                feats = self.net.feature_extractor.conv3x3_ocr(feats)

                context = self.net.feature_extractor.ocr_gather_head(feats, out_aux)
                backbone_features = self.net.feature_extractor.ocr_distri_head(feats, context)
            else:
                raise NotImplementedError

        return backbone_features


class InputBRSPredictor(BRSBasePredictor):
    def __init__(self, model, device, opt_functor, optimize_target='rgb', **kwargs):
        super().__init__(model, device, opt_functor=opt_functor, **kwargs)
        self.optimize_target = optimize_target

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])
        num_clicks = len(clicks_lists[0])

        if self.opt_data is None or is_image_changed:
            opt_channels = 2 if self.optimize_target == 'dmaps' else 3
            bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]
            self.opt_data = torch.zeros((bs, opt_channels, image_nd.shape[2], image_nd.shape[3]),
                                        device=self.device, dtype=torch.float32)

        def get_prediction_logits(opt_bias):
            input_image = image_nd
            if self.optimize_target == 'rgb':
                input_image = input_image + opt_bias
            dmaps = self.net.dist_maps(input_image, points_nd)
            if self.optimize_target == 'dmaps':
                dmaps = dmaps + opt_bias

            x = self.net.rgb_conv(torch.cat((input_image, dmaps), dim=1))
            if self.optimize_target == 'all':
                x = x + opt_bias

            if isinstance(self.net, DistMapsHRNetModel):
                pred_logits = self.net.feature_extractor(x)[0]
            else:
                backbone_features = self.net.feature_extractor(x)
                pred_logits = self.net.head(backbone_features[0])
            pred_logits = F.interpolate(pred_logits, size=image_nd.size()[2:], mode='bilinear', align_corners=True)

            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask, self.device,
                                    shape=self.opt_data.shape)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data.cpu().numpy().ravel(),
                                       **self.opt_functor.optimizer_params)

            self.opt_data = torch.from_numpy(opt_result[0]).view(self.opt_data.shape).to(self.device)

        with torch.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_vars, _ = self.opt_functor.unpack_opt_params(self.opt_data)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits
    
    
##########################################################    
    
from .cgam_module import CGAM

class CGAMPredictor(BRSBasePredictor):
    def __init__(self, model, device, opt_functor, insertion_mode='after_deeplab', **kwargs):
        super().__init__(model, device, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
        self._c1_features = None
        
        self.newest_click_mask = None
        self.click4agm = None
        self.click_maps = None
        self.is_positive = None
        
        self.att_mask = None
        self.click_maps_vis = None
        
        self.newest_click_mask_vis = None
        self.is_positive_vis = None
        
        self.early_stopping = False

        if self.insertion_mode == 'after_deeplab':
            self.num_channels = model.feature_extractor.ch
        elif self.insertion_mode == 'after_c4':
            self.num_channels = model.feature_extractor.aspp_in_channels
        elif self.insertion_mode == 'after_aspp':
            self.num_channels = model.feature_extractor.ch + 32
        else:
            raise NotImplementedError
        
        self.cgam = CGAM()

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])

        num_clicks = len(clicks_lists[0])

        if num_clicks <= self.net_clicks_limit or is_image_changed or self.input_data is None:
            self.input_data = self._get_head_input(image_nd, points_nd)
            if num_clicks == 1:
                self.cgam.initialize(image_nd.shape, self.input_data.shape)
                self.cgam.to(self.device)
                self.opt_functor.prev_logits = None
         
        def get_prediction_logits(aux_layer, num_clicks):
            if num_clicks == 1:
                scaled_backbone_features = self.input_data
            else:
                scaled_backbone_features = aux_layer(self.input_data, self.click_maps)
            
            if self.insertion_mode == 'after_c4':
                x = self.net.feature_extractor.aspp(scaled_backbone_features)
                x = F.interpolate(x, mode='bilinear', size=self._c1_features.size()[2:],
                                  align_corners=True)
                x = torch.cat((x, self._c1_features), dim=1)
                scaled_backbone_features = self.net.feature_extractor.head(x)
            elif self.insertion_mode == 'after_aspp':
                scaled_backbone_features = self.net.feature_extractor.head(scaled_backbone_features)

            pred_logits = self.net.head(scaled_backbone_features)
            pred_logits = F.interpolate(pred_logits, size=image_nd.size()[2:], mode='bilinear',
                                        align_corners=True)
            return pred_logits

        if num_clicks > self.optimize_after_n_clicks:
            if num_clicks == self.optimize_after_n_clicks+1:
                self.opt_functor.init_optimizer(self, self.device)
                iteration = 20
            else:
                iteration = 20
            with torch.enable_grad():
                self.newest_click_mask, self.click4cgam, radius, self.is_positive  = self._get_newest_click_map(clicks_lists, image_nd.shape[2:])
                self.click_maps = self._get_CGAM_click_maps(clicks_lists, image_nd.shape[2:])
                
                for i in range(iteration):
                    pred_logits = get_prediction_logits(self.cgam, num_clicks)
                    
                    if i == 0:
                        self.opt_functor.prev_logits = pred_logits.detach()
                    
                    self.early_stopping = self.opt_functor.optimize(pred_logits, pos_mask, neg_mask, self.newest_click_mask, i)
                    if self.early_stopping:
                        break
    
        with torch.no_grad():
            opt_pred_logits = get_prediction_logits(self.cgam, num_clicks)
        return opt_pred_logits
    
    def _get_prediction_with_result(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])

        num_clicks = len(clicks_lists[0])

        if num_clicks <= self.net_clicks_limit or is_image_changed or self.input_data is None:
            self.input_data = self._get_head_input(image_nd, points_nd)
            if num_clicks == 1:
                self.cgam.initialize(image_nd.shape, self.input_data.shape)
                self.cgam.to(self.device)
                self.opt_functor.prev_logits = None
         
        def get_prediction_logits(aux_layer, num_clicks):
            if num_clicks == 1:
                scaled_backbone_features = self.input_data
            else:
                scaled_backbone_features, self.att_mask = aux_layer(self.input_data, self.click_maps)
            
            if self.insertion_mode == 'after_c4':
                x = self.net.feature_extractor.aspp(scaled_backbone_features)
                x = F.interpolate(x, mode='bilinear', size=self._c1_features.size()[2:],
                                  align_corners=True)
                x = torch.cat((x, self._c1_features), dim=1)
                scaled_backbone_features = self.net.feature_extractor.head(x)
            elif self.insertion_mode == 'after_aspp':
                scaled_backbone_features = self.net.feature_extractor.head(scaled_backbone_features)

            pred_logits = self.net.head(scaled_backbone_features)
            pred_logits = F.interpolate(pred_logits, size=image_nd.size()[2:], mode='bilinear',
                                        align_corners=True)
            return pred_logits

        if num_clicks > self.optimize_after_n_clicks:
            if num_clicks == self.optimize_after_n_clicks+1:
                self.opt_functor.init_optimizer(self, self.device)
                iteration = 20
            else:
                iteration = 20
            with torch.enable_grad():
                self.newest_click_mask, self.click4cgam, radius, self.is_positive  = self._get_newest_click_map(clicks_lists, image_nd.shape[2:])
                self.newest_click_mask_vis, _, _, self.is_positive_vis  = self._get_newest_click_map_vis(clicks_lists, image_nd.shape[2:])
                self.click_maps = self._get_CGAM_click_maps(clicks_lists, image_nd.shape[2:])
                self.click_maps_vis = self._get_click_maps_vis(clicks_lists, image_nd.shape[2:])
                
                for i in range(iteration):
                    pred_logits = get_prediction_logits(self.cgam, num_clicks)
                    
                    if i == 0:
                        self.opt_functor.prev_logits = pred_logits.detach()
                    
                    self.early_stopping = self.opt_functor.optimize(pred_logits, pos_mask, neg_mask, self.newest_click_mask, i)
                    if self.early_stopping:
                        break
            
        with torch.no_grad():
            opt_pred_logits = get_prediction_logits(self.cgam, num_clicks)
        
        click_maps_vis  = self._get_click_maps_vis(clicks_lists, image_nd.shape[2:])
        
        return opt_pred_logits, self.click_maps_vis, self.click_maps_vis, self.att_mask, self.newest_click_mask_vis, self.is_positive_vis

    def _get_head_input(self, image_nd, points):
        with torch.no_grad():
            coord_features = self.net.dist_maps(image_nd, points)
            x = self.net.rgb_conv(torch.cat((image_nd, coord_features), dim=1))
            if self.insertion_mode == 'after_c4' or self.insertion_mode == 'after_aspp':
                c1, _, c3, c4 = self.net.feature_extractor.backbone(x)
                c1 = self.net.feature_extractor.skip_project(c1)

                if self.insertion_mode == 'after_aspp':
                    x = self.net.feature_extractor.aspp(c4)
                    x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=True)
                    x = torch.cat((x, c1), dim=1)
                    backbone_features = x
                else:
                    backbone_features = c4
                    self._c1_features = c1
            else:
                backbone_features = self.net.feature_extractor(x)[0]

        return backbone_features
