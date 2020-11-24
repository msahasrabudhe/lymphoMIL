"""
Model redefined for lymphocytosis. 

Major changes:
    1.      Training and model code is different.
    2.      Move from command-line arguments to YAML configuration files. 
    3.      Code restructing and reorganising to make it easier to follow
            as well as restrict repeat code.
"""

# ===============================================================================================================================
#   All architectures needed for this model.
from    helper_models       import  *

# ===============================================================================================================================
#   Dataset definitions
from    datasets            import  *

# ===============================================================================================================================
#   Torch imports
import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      F
import  torch.optim         as      optim
from    torch.autograd      import  Variable, Function

# ===============================================================================================================================
#   System imports
import  sys
import  os


# ===============================================================================================================================
#   Scikit-learn imports
from    sklearn             import  metrics


# ===============================================================================================================================
#   Image-related imports
import  imageio

# ===============================================================================================================================
#   Logging-related imports
import  matplotlib
import  matplotlib.pyplot   as      plt

# ===============================================================================================================================
#   Debugging imports
from    IPython             import  embed


# ===============================================================================================================================
#   Other imports
from    attr_dict           import  *


# ===============================================================================================================================
#   Global variables

# Aggregation strategies. 
MAX                                         = 'max'
AVG                                         = 'avg'
NAND                                        = 'nand'
LME                                         = 'lme'
LSE                                         = 'lse'
IMPLEMENTED_AGGREGATION_MODES               = [MAX, AVG, LME, LSE]

INV_SQRT_2PI                                = 1./np.sqrt(2 * np.pi).astype(np.float32)

# ===============================================================================================================================
#   The CNN-mixer-Gating, all-in-one model redefined as an nn.Module
class Model(nn.Module):
    def __init__(self, options=None):
        """
        Init function takes in an attr_dict or a string.
        If it is an attr_dict, a model is built based on this dict. 
        If it is a string and points to a directory, a model is built based on from os.path.join(options, 'options.pkl')
        If it is a string and points to a .yaml file, a model is built according to that yaml file. 
        If options is None, a template configuration attr_dict is picked up from configs/template.yaml
        """
        super(Model, self).__init__()

        if options is None:
            print(cfm('INFO', 'y', 'b') + ': No options specified. Loading from default .yaml --- configs/template.yaml.')
            options                         = load_yaml('configs/template.yaml') 
            options                         = fix_backward_compatibility(options)
        elif not (isinstance(options, str) or isinstance(options, AttrDict)):
            print(cfm('HALT', 'r', 'b') + ': Specified variable options is neither a string, nor an AttrDict.')
            raise ValueError
        elif isinstance(options, str):
            if os.path.isdir(options):
                with open(os.path.join(options, 'options.pkl'), 'rb') as fp:
                    options                 = pickle.load(fp)
            elif options.endswith('.yaml'):
                options                     = load_yaml(options)


        self.system_mode                    = options.model.system_mode
        if self.system_mode not in SYSTEM_MODES:
            ValueError('Specified system mode [%s] unrecognised! Permitted values are [%s].' %(self.system_mode, ', '.join(SYSTEM_MODES)))
        if 'D' in self.system_mode and 'I' not in self.system_mode:
            ValueError('Decoder can only be trained when an encoder is also being used (please use \'I\' with \'D\' in model.system_mode)!')
                    

        # =======================================================================================================================
        #   Architecture options.
        self.arch                           = options.model.arch
        self.agg_mode                       = options.model.agg_mode
        self.test_agg_mode                  = options.model.test_agg_mode
        self.r_lme                          = options.model.r_lme
        self.r_lse                          = options.model.r_lse
        self.ndf                            = options.model.ndf
        self.score                          = options.model.score
        self.mix                            = options.model.mix 
        self.use_encoder                    = options.model.use_encoder
        self.use_decoder                    = options.model.use_decoder
        self.predictor_type                 = options.model.predictor_type
        self.attention_arch                 = options.model.attention_arch
        self.separate_gate_fex              = options.model.separate_gate_fex
        rho                                 = torch.FloatTensor([[[options.model.rho]]])
        self.register_buffer('rho', rho)

        #   Attributes to use. 
        self.attr_to_use                    = options.model.attr_to_use
        for _at in self.attr_to_use:
            if _at not in ACCEPTED_PATIENT_ATTRIBUTES:
                print('Attribute [%s] not accepted currently for training!' %(_at))
                raise ValueError
        #   The number of inputs to the MLP classifier is thus the length of this vector. 
        options.model.mlp_nin               = len(self.attr_to_use)
        self.mlp_nin                        = options.model.mlp_nin
        #   Also create a list of statistics for attributes. 
        #   The order of this is the same as the one in options.model.attr_to_use
        self.attr_means                     = options.training.attr_means
        self.attr_stds                      = options.training.attr_stds
        #   Sanity checks. 
        if len(self.attr_means) != self.mlp_nin:
            print('The number of means (%d) specified for attributes does not match the number of attributes (%d).' 
                    %(len(self.attr_means), self.mlp_nin))
        if len(self.attr_stds) != self.mlp_nin:
            print('The number of STDs (%d) specified for attributes does not match the number of attributes (%d).' 
                    %(len(self.attr_stds), self.mlp_nin))
        

        if self.agg_mode not in IMPLEMENTED_AGGREGATION_MODES:
            print('Aggregation strategy [%s] not implemented! Please choose one of [%s].' %(self.agg_mode, ', '.join(IMPLEMENTED_AGGREGATION_MODES)))
            raise NotImplementedError
        if self.test_agg_mode not in IMPLEMENTED_AGGREGATION_MODES:
            print('Aggregation strategy [%s] not implemented! Please choose one of [%s].' %(self.test_agg_mode, ', '.join(IMPLEMENTED_AGGREGATION_MODES)))
            raise NotImplementedError

        # =======================================================================================================================
        #   Training options. 
        self.loss_to_use                    = options.training.loss
        if self.loss_to_use not in IMPLEMENTED_LOSSES:
            print('Loss [%s] not yet implemented!' %(self.loss_to_use))
            raise ValueError

        self.img_size                       = options.training.img_size
        self.imgs_batch_size                = options.training.imgs_batch_size
        self.subj_batch_size                = options.training.subj_batch_size
        self.use_cuda                       = options.training.cuda
        self.lr                             = options.training.lr
        self.lr_decay                       = options.training.lr_decay
        self.weight_decay                   = options.training.weight_decay
        self.beta1                          = options.training.beta1
        self.optim_alg                      = options.training.optim_alg
        self.recon_fn                       = options.training.recon_fn
        self.weights_rescale                = 1


        self.register_buffer('dummy', torch.FloatTensor([0]))
        self.dummy.requires_grad=True

        # =======================================================================================================================
        #   Dataset options
        self.img_means                      = options.training.img_means
        self.img_stds                       = options.training.img_stds
        self.img_augment                    = options.training.augment
        self.freeze_cnn                     = options.training.freeze_cnn
        self.freeze_mlp                     = options.training.freeze_mlp
        self.img_ext                        = options.dataset.img_ext.lower()
        self.c_transform                    = options.dataset.c_transform
        #   Sanity checks for img_means and img_stds. 
        if len(self.img_means) != options.model.nc:
            print('The number of means specified for images (%d) does not match the number of input channels (%d). '
                    %(len(self.img_means), options.model.nc)) 
        if len(self.img_stds) != options.model.nc:
            print('The number of STDs specified for images (%d) does not match the number of input channels (%d). '
                    %(len(self.img_stds), options.model.nc)) 

        # =======================================================================================================================
        #   Other options
        self.experiment_name                = options.experiment_name
        self.save_dir                       = options.output_dir

        self.cpu_map_location               = lambda storage, location: storage

        # =======================================================================================================================
        #   Model definitions.
        self.models_dict                    = {}
        if 'I' in self.system_mode:
            # Create CNN
            self.cnn                        = make_cnn(options)

            self.models_dict['cnn']         = self.cnn

        if 'N' in self.system_mode:
            self.attention                  = make_attention_network(options, self.cnn.out_vec_shape)
            self.models_dict['attention']   = self.attention

        if self.use_encoder:
            self.encoder                    = nn.Sequential(
                                                conv3x3(self.cnn.out_chs, self.cnn.out_chs, stride=1),
                                                nn.BatchNorm2d(self.cnn.out_chs),
                                                nn.Tanh(),
                                              )
            self.models_dict['encoder']     = self.encoder
        else:
            self.encoder                    = Identity()

        if 'A' in self.system_mode:
            # Create classifier
            self.mlp                        = MLP(options)

            self.models_dict['mlp']         = self.mlp

        if 'G' in self.system_mode or 'M' in self.system_mode:
            assert(not self.score)
            # Create Gate. 
            aps                             = int(self.cnn.out_shape)

            if self.separate_gate_fex:
                self.gate_fex               = make_cnn(options)
#   old version, with avg_pool    
#            self.avg_pool                   = nn.AvgPool2d((aps, aps))
#            self.gate                       = nn.Linear(self.cnn.out_chs + options.model.mlp_nin, 1)

# new version, with full conv
            if options.model.gating_network == 'softmax':
                if self.predictor_type == 'avg_pool':
                    self.gate               = GatingNetworkAvgPool(aps, self.cnn.out_chs, options.model.mlp_nin)
                else:
                    self.gate               = GatingNetwork(aps, self.cnn.out_chs, options.model.mlp_nin)
            elif options.model.gating_network == 'sigmoid':
                self.gate                   = GatingNetworkSigmoid(aps, self.cnn.out_chs, options.model.mlp_nin)
            elif options.model.gating_network == 'concat_avg-pool_softmax':
                self.gate                   = GatingNetworkConcatAvgPool(aps, self.cnn.out_chs, options.model.mlp_nin)
            else:
                raise ValueError('Invalid gating_network type. Expected one of [softmax, sigmoid]. Got {}'.format(options.model.gating_network))

            self.models_dict['gate']        = self.gate

        if 'D' in self.system_mode:
            # Create decoder.
            options.model.cnn_out_shape     = self.cnn.out_shape
            self.decoder                    = UpsampleDecoder(options)

            self.models_dict['decoder']     = self.decoder

        if not self.score and 'I' in self.system_mode:
            # Create classifier. 
            
            # The following concatenates the attributs to the images feature vector. 
            # should only be used with full_conv when the conv backbone outputs a flat feature vector,
            # or with avg_pool
            extra_attr_concat_feats         = ('C' in self.system_mode) * len(options.model.attr_to_use)
            print(extra_attr_concat_feats)

            if 'N' not in self.system_mode:
                if self.predictor_type.startswith('fc'):
                    h_units                 = [int(x) for x in self.predictor_type.split('_')[1].split('-')]
                    clf_layers              = [ View([-1, self.cnn.out_vec_shape]) ]
                    prev_h                  = self.cnn.out_vec_shape
                    for h in h_units:
                        clf_layers.append( nn.Linear(prev_h, h) )
                        clf_layers.append( nn.ReLU(True) )
                        clf_layers.append( nn.Dropout() )
                        prev_h              = h
                    clf_layers.append( nn.Linear(h, self.cnn.n_classes) )
                    # Add extra dimensions to make compatible with the convolutional predictor types. 
                    clf_layers.append( View([-1, self.cnn.n_classes, 1, 1]) )
                    self.classifier         = nn.Sequential( *clf_layers )

                elif self.predictor_type == 'multi_conv':
                    self.classifier         = nn.Sequential(
                            nn.Conv2d(self.cnn.out_chs, self.cnn.out_chs // 2, kernel_size=5, stride=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(self.cnn.out_chs // 2, self.cnn.n_classes, kernel_size=3, stride=1, padding=0),
                    )
            
                elif self.predictor_type == 'multi_conv_3':
                    self.classifier         = nn.Sequential(
                            nn.Conv2d(self.cnn.out_chs, self.cnn.out_chs // 8, kernel_size=3, stride=2, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(self.cnn.out_chs // 8, self.cnn.n_classes, kernel_size=3, stride=1, padding=0),
                    )

                elif self.predictor_type == 'full_conv':
                    self.classifier         = nn.Conv2d(self.cnn.out_chs + extra_attr_concat_feats, self.cnn.n_classes, self.cnn.out_shape, stride=1, padding=0)

                elif self.predictor_type == 'avg_pool':
                    self.classifier         = nn.Sequential(
                                                nn.AdaptiveAvgPool2d((1, 1)),
                                                nn.Conv2d(self.cnn.out_chs + extra_attr_concat_feats, self.cnn.n_classes, kernel_size=1, stride=1, padding=0),
                                              )

                elif self.predictor_type == 'max_pool':
                    self.classifier         = nn.Sequential(
                                                nn.MaxPool2d(self.cnn.out_shape),
                                                nn.Conv2d(self.cnn.out_chs, self.cnn.n_classes, kernel_size=1, stride=1, padding=0),
                                              )
                else:
                    raise ValueError(cfm('HALT', 'r', 'b') + ': predictor_type must be one of \'fc\', \'multi_conv\', \'full_conv\', \'avg_pool\', \'max_pool\' (got %s).' %(self.predictor_type))
            else:
                self.classifier             = nn.Linear(self.attention.L * self.attention.K, 1, bias=False)

            self.models_dict['classifier']  = self.classifier
        
        # =======================================================================================================================
        #   Optimiser definitions
        self.optimisers_dict                = {}

        if 'cnn' in self.models_dict and not self.freeze_cnn:
            self.cnn_optimiser              = self.create_optimiser(
                                                self.cnn.parameters(), 
                                                scale=options.training.scale_cnn,
                                              )
            self.optimisers_dict['cnn']     = self.cnn_optimiser

        if 'attention' in self.models_dict:
            self.attention_optimiser        = self.create_optimiser(
                                                self.attention.parameters(),
                                                scale=options.training.scale_cnn,
                                              )
            self.optimisers_dict['attention']   = self.attention_optimiser

        if self.use_encoder:
            self.encoder_optimiser          = self.create_optimiser(
                                                self.cnn.parameters(),
                                                scale=options.training.scale_cnn,
                                              )

            self.optimisers_dict['encoder'] = self.encoder_optimiser

        if 'mlp' in self.models_dict and not self.freeze_mlp:
            self.mlp_optimiser              = self.create_optimiser(
                                                self.mlp.parameters(),
                                                scale=options.training.scale_mlp,
                                              )
            self.optimisers_dict['mlp']     = self.mlp_optimiser

        if 'gate' in self.models_dict:
            self.gate_optimiser             = self.create_optimiser(
                                                self.gate.parameters(),
                                                scale=options.training.scale_gate,
                                              )
            self.optimisers_dict['gate']    = self.gate_optimiser

        if 'decoder' in self.models_dict:
            self.decoder_optimiser          = self.create_optimiser(
                                                self.decoder.parameters(),
                                                scale=options.training.scale_decoder,
                                              )
            self.optimisers_dict['decoder'] = self.decoder_optimiser

        if 'classifier' in self.models_dict and not self.freeze_cnn:
            self.classifier_optimiser       = self.create_optimiser(
                                                self.classifier.parameters(),
                                                scale=options.training.scale_classifier,
                                              )
    # ===========================================================================================================================


    def create_optimiser(self, params, scale=1):
        if self.optim_alg == 'sgd':
            return optim.SGD(
                    params, 
                    lr=scale * self.lr,
                    weight_decay=self.weight_decay,
                    momentum=self.beta1,
                   )
        elif self.optim_alg == 'adam':
            return optim.Adam(
                    params, 
                    lr=scale * self.lr, 
                    weight_decay=self.weight_decay,
                    betas=(self.beta1, 0.999),
                   )
        else:
            raise ValueError('optim_alg must be either sgd or adam.')

    # ===========================================================================================================================


    def scores_aggregator(self, p_img_scores, test=False, weights=None):
        agg_mode                            = self.test_agg_mode if test else self.agg_mode

        if weights is None:
            weights                         = 1
        else:
            assert weights.size(0) == p_img_scores.size(0), \
                'Dimension 0 for p_img_scores(%d) and weights must be of the same size in Model.scores_aggregator.' %(p_img_scores.size(0), weights.size(0))

            ndim                            = p_img_scores.dim()
            for __e in range(ndim-1):
                weights                     = weights.unsqueeze(-1)

        if agg_mode == MAX:
            agg_score                       = torch.max(weights * p_img_scores, dim=0, keepdim=True)[0]

        elif agg_mode == AVG:
            agg_score                       = torch.mean(weights * p_img_scores, dim=0, keepdim=True)

        elif agg_mode == LME:
            agg_score                       = 1./ self.r_lme * \
                                              torch.log(
                                                torch.mean(
                                                    torch.exp(
                                                        self.r_lme * \
                                                        weights * \
                                                        p_img_scores
                                                    ),
                                                    dim=0,
                                                    keepdim=True,
                                                )
                                              )

        elif agg_mode == LSE:
            agg_score                       = 1./ self.r_lse * \
                                              torch.log(
                                                torch.sum(
                                                    torch.exp(
                                                        self.r_lse * \
                                                        weights * \
                                                        p_img_scores
                                                    ),
                                                    dim=0,
                                                    keepdim=True,
                                                )
                                              )
        return agg_score
    # ===========================================================================================================================

                                                    

    def reset_gradients(self):
        """
        Reset all parameter gradients.
        """
        
        for optim_key in self.optimisers_dict:
            self.optimisers_dict[optim_key].zero_grad()
        return
    # ===========================================================================================================================



    def take_optimiser_step(self):
        """
        Take optimiser step for all optimisers.
        """

        for optim_key in self.optimisers_dict:
            self.optimisers_dict[optim_key].step()
        return
    # ===========================================================================================================================


    def gate_fex_forward(self, images):
        """
        Forward function for the feature extractor for gating network.
        """
        if self.freeze_cnn:
            self.gate_fex.eval()
        results                                 = self.gate_fex(images)
        return results
    # ===========================================================================================================================


    def cnn_forward(self, images):
        """
        Forward function for the cnn
        """
        if self.freeze_cnn:
            self.cnn.eval()
        results                                 = self.cnn(images)
        return results
    # ===========================================================================================================================


    def mlp_forward(self, attributes):
        """
        Forward function for the MLP.
        """
        if self.freeze_mlp:
            self.mlp.eval()
        results                                 = self.mlp(attributes).squeeze(1)
        return results
    # ===========================================================================================================================


    def decoder_forward(self, images):
        """
        Forward function for the decoder. 
        """
        results                                 = self.decoder(images)
        return results
    # ===========================================================================================================================


    def forward(self, p_root, p_attr, p_mask=False, test=False, ae_only=False, gradcam=False):
        """
        Forward on a patient, given the path to images. 

        Inputs
            p_root              Path to the directory containing all images for this patient
            p_attr              A dictionary containing tensors for all attributes. 
        """

        return_dict                             = {}

   
        with ExitStack() as stack:
            if test and not gradcam:
                stack.enter_context(torch.no_grad())

            if 'cnn' in self.models_dict:
                # If we are supposed to train a CNN on images. Otherwise skip this step.
                # At the end of this step, we will have a variable p_img_scores, which
                #   will record the CNN's score for all patient images. 
    
                # Patient dataset. 
                if p_mask:
                    p_dataset                   = PatientNormMaskedImageSet(
                                                    p_root, 
                                                    p_mask,
                                                    transform=False if test else self.img_augment,
                                                    mean=self.img_means,
                                                    std=self.img_stds,
                                                    ext=[self.img_ext],
                                                    c_transform=self.c_transform,
                                                    test=test,
                                                  )
                else:
                    p_dataset                   = PatientNormImageSet(
                                                    p_root, 
                                                    transform=False if test else self.img_augment,
                                                    mean=self.img_means,
                                                    std=self.img_stds,
                                                    ext=[self.img_ext],
                                                    c_transform=self.c_transform,
                                                    test=test,
                                                  )
    
                # Patient dataloader. 
                p_dataloader                    = torch.utils.data.DataLoader(
                                                    p_dataset,
                                                    shuffle=not test,
                                                    batch_size=self.imgs_batch_size,
                                                    num_workers=8
                                                  )
    
                # Compute scores for all iamges
                for batch_idx, p_data in enumerate(p_dataloader):
                    if self.use_cuda:
                        p_data                  = p_data.cuda()

                    if gradcam:
                        p_data.requires_grad    = True
                        p_images                = p_data if batch_idx == 0 else torch.cat([p_images, p_data], dim=0)
    
                    p_img_result_               = self.cnn_forward(p_data)
                    if self.separate_gate_fex:
                        p_img_gate_fex_         = self.gate_fex(p_data)['latent']
                        p_img_gate_fex          = p_img_gate_fex_ if batch_idx == 0 else torch.cat((p_img_gate_fex, p_img_gate_fex_), dim=0)
                         

                    # Compute the sizes of the lymphocytes. 
                    mask                        = (p_data.sum(dim=1).unsqueeze(1)) > 0
                    cover                       = mask.view(p_data.size(0), -1).sum(dim=1).float()
                    weights_                    = cover / (self.img_size * self.img_size)

                    if self.score:
                        p_img_scores_           = p_img_result_['score']
                        if batch_idx == 0:
                            p_img_scores        = p_img_scores_
                        else:
                            p_img_scores        = torch.cat((
                                                        p_img_scores,
                                                        p_img_scores_,
                                                        ),
                                                    dim=0,
                                                  )
            
                    p_img_latent_               = p_img_result_['latent']
                    p_img_latent_enc_           = self.encoder(p_img_latent_)
                    if batch_idx == 0:
                        p_img_latent            = p_img_latent_
                        p_img_latent_enc        = p_img_latent_enc_
                        p_data_stack            = p_data
                        weights                 = weights_
                    else:
                        p_img_latent            = torch.cat((
                                                        p_img_latent, 
                                                        p_img_latent_,
                                                        ),
                                                    dim=0,
                                                  )
                        p_img_latent_enc        = torch.cat((
                                                        p_img_latent_enc, 
                                                        p_img_latent_enc_,
                                                        ),
                                                    dim=0,
                                                  )
                        p_data_stack            = torch.cat((
                                                        p_data_stack,
                                                        p_data,
                                                        ),
                                                    dim=0,
                                                  )
                        weights                 = torch.cat((
                                                        weights, 
                                                        weights_,
                                                        ),
                                                    dim=0,
                                                  )

                    # Train decoder. 
                    if 'D' in self.system_mode:
                        p_img_recon_            = self.decoder_forward(p_img_latent_enc_)

                        if self.recon_fn == 'mse':
                            loss_recon_         = F.mse_loss(p_img_recon_, p_data, reduce=True, size_average=False)
                        elif self.recon_fn == 'l1':
                            loss_recon_         = F.l1_loss(p_img_recon_, p_data, reduce=True, size_average=False)
                        elif self.recon_fn == 'cc':
                            loss_recon_         = cross_correlation_loss(p_data, p_img_recon_, n=9)
                        else:
                            raise ValueError('Unrecognised reconstruction loss function (%s).' %(self.recon_fn))

                        # Get gradients from this loss here because we want to save memory. 
                        if not test and ae_only:
                            loss_recon_.backward()

                        if batch_idx == 0:
                            p_img_recon         = p_img_recon_

                            loss_recon          = loss_recon_
                        else:
                            p_img_recon         = torch.cat((
                                                        p_img_recon,
                                                        p_img_recon_,
                                                        ),
                                                    dim=0,
                                                  )

                            loss_recon          = loss_recon + loss_recon_

                    
#                weights                         = self.weights_rescale * weights
                weights                         = None
                # Aggregate scores. 
                if self.score:
                    p_img_scores                = p_img_scores.squeeze(1)
                    p_img_score_agg             = self.scores_aggregator(p_img_scores, test=test, weights=weights)
                else:
                       

                    
                    if 'N' in self.system_mode:
                        p_img_latent_flat       = p_img_latent.view(-1, self.cnn.out_vec_shape)
                        p_img_scores            = self.classifier(self.attention.H_net(p_img_latent_flat)).squeeze(1)
                        p_img_attention_res     = self.attention(p_img_latent_flat)
                        p_img_latent_agg        = p_img_attention_res[0]
                        p_img_latent_weights    = p_img_attention_res[1]
                        p_img_score_agg         = self.classifier(p_img_latent_agg).squeeze(1)
                        return_dict['p_img_latent_weights'] = p_img_latent_weights
                    else:
                        # using p_img_latent instead of p_img_latent_enc everywhere. 

                        # If 'C' is in system mode, append the attributes. 
                        if 'C' in self.system_mode:
                            # Normalise p_attr
                            for _i, _attr in enumerate(self.attr_to_use, 0):
                                p_attr[_attr]               = (p_attr[_attr] - self.attr_means[_i]) / self.attr_stds[_i]

                            # Create the input to the MLP. 
                            p_attr_input                    = torch.cat([p_attr[_attr] for _attr in self.attr_to_use], dim=1)
                            # Also unsqueeze dimensions 2 and 3
                            p_attr_input                    = p_attr_input.unsqueeze(2).unsqueeze(3)
            
                            # If we are training on attributes as well.
                            if self.use_cuda:
                                p_attr_input                = p_attr_input.cuda()
                            _, NA_, _, _                    = p_attr_input.size()

                            
                        if 'C' in self.system_mode: 
                            p_img_scores        = self.classifier(
                                                    torch.cat([p_img_latent, 
                                                               p_attr_input.expand(
                                                                   [p_img_latent.shape[0],NA_,1,1]
                                                               )], 
                                                              dim=1)).squeeze(-1).squeeze(-1).squeeze(1)
                        else:
                            p_img_scores        = self.classifier(p_img_latent).squeeze(-1).squeeze(-1).squeeze(1)

                        p_img_latent_agg        = self.scores_aggregator(p_img_latent, test=test, weights=weights)
                        
                        if 'C' in self.system_mode:
                            p_img_score_agg     = self.classifier(torch.cat([p_img_latent_agg, p_attr_input], dim=1)).squeeze(-1).squeeze(-1).squeeze(1)
                        else:
                            p_img_score_agg     = self.classifier(p_img_latent_agg).squeeze(-1).squeeze(-1).squeeze(1)

                if self.separate_gate_fex:
                    gate_fex_pooled             = self.scores_aggregator(p_img_gate_fex, test=test, weights=None)
                else:
                    gate_fex_pooled             = p_img_latent_agg

                # Add to return dict. 
                return_dict['img_scores']       = p_img_scores
                return_dict['agg_img_score']    = p_img_score_agg
                return_dict['p_img_latent']     = p_img_latent
                return_dict['p_img_latent_enc'] = p_img_latent_enc
                return_dict['img_inputs']       = p_data_stack

                if gradcam:
                    return_dict['p_images']     = p_images


                if 'D' in self.system_mode:
                    # Divide loss_recon by the number of images for this patient. 
                    loss_recon                  = loss_recon / len(p_dataset) / self.img_size / self.img_size
#                    rho_hat                     = p_img_latent_enc.mean(dim=0)
                    loss_sparse                 = F.tanh(self.dummy)
                    # Add images and loss to return dict
                    return_dict['img_recons']   = p_img_recon
                    return_dict['loss_recon']   = loss_recon
                    return_dict['loss_sparse']  = loss_sparse

            if 'mlp' in self.models_dict:
                # Normalise p_attr
                for _i, _attr in enumerate(self.attr_to_use, 0):
                    p_attr[_attr]               = (p_attr[_attr] - self.attr_means[_i]) / self.attr_stds[_i]

                # Create the input to the MLP. 
                p_attr_input                    = torch.cat([p_attr[_attr] for _attr in self.attr_to_use], dim=1)

                # If we are training on attributes as well.
                if self.use_cuda:
                    p_attr_input                = p_attr_input.cuda()
                p_attr_score                    = self.mlp_forward(p_attr_input)

                # Add this to return dict. 
                return_dict['attr_score']       = p_attr_score

            if 'G' in self.system_mode or 'M' in self.system_mode:
                assert('mlp' in self.models_dict and 'cnn' in self.models_dict)
# old version, with avg_pool
#                p_img_latent_pooled             = self.avg_pool(p_img_latent_agg).squeeze(-1).squeeze(-1)
#                p_gate_input                    = torch.cat((p_img_latent_pooled, p_attr_input), dim=1)
#                p_gate_pred                     = F.sigmoid(self.gate(p_gate_input))

# new version, with full conv
                p_gate_pred                     = self.gate(gate_fex_pooled, p_attr_input)

                return_dict['w_img_score']      = p_gate_pred

            if self.system_mode in [IMGS_TR, IMGS_CONCAT_ATTR_TR, IMGS_AUTOENC_TR, IMGS_ATTN_TR]:
                return_dict['p_score']          = p_img_score_agg
            elif self.system_mode == ATTR_TR:
                return_dict['p_score']          = p_attr_score
            elif self.system_mode in [IMGS_ATTR_TR, IMGS_AUTOENC_ATTR_TR]:
                return_dict['p_score']          = 0.5 * p_img_score_agg + 0.5 * p_attr_score
            elif self.system_mode in [IMGS_ATTR_GATE_TR, IMGS_ATTR_GATE_GATELOSS_TR, IMGS_ATTR_GATE_MOG_TR, IMGS_ATTR_GATE_MSE_TR]:
                return_dict['p_score']          = p_gate_pred[:,0] * F.sigmoid(p_img_score_agg) + (1 - p_gate_pred[:,0]) * F.sigmoid(p_attr_score)
   
        # Return the result.  
        return return_dict 
    # ===========================================================================================================================


   
    def checkpoint(self, epoch=None, best=False):
        """
        Checkpoint this model at the location defined by save_dir.
        """

        # Create a directory for the best model first. 
        if not os.path.exists(os.path.join(self.save_dir, 'best_model/')):
            os.makedirs(os.path.join(self.save_dir, 'best_model/'))

        # Save models first. 
        __saved_models                          = {}
        for __mname in self.models_dict:
            __saved_models[__mname]             = self.models_dict[__mname].state_dict()
        if best:
            save_path                           = os.path.join(self.save_dir, 'best_model/', 'nets.pth')
        else:
            save_path                           = os.path.join(self.save_dir, 'nets.pth')
        torch.save(__saved_models, save_path)

        # Save optimisers now
        __saved_optimisers                      = {}
        for __oname in self.optimisers_dict:
            __saved_optimisers[__oname]         = self.optimisers_dict[__oname].state_dict()
        if best:
            save_path                           = os.path.join(self.save_dir, 'best_model/', 'optimisers.pth')
        else:
            save_path                           = os.path.join(self.save_dir, 'optimisers.pth')
        torch.save(__saved_optimisers, save_path)
        
        # Save models and optimisers explicitly for this epoch as well if it is specified.
        if epoch is not None:
            save_path                           = os.path.join(self.save_dir, '%d_nets.pth' %(epoch))
            torch.save(__saved_models, save_path)
            save_path                           = os.path.join(self.save_dir, '%d_optimisers.pth' %(epoch))
            torch.save(__saved_optimisers, save_path)

        print(cfm('SAVE', 'g', 'b') + ': Saved models to %s.' %(self.save_dir))
        if best:
            print(cfm('SAVE', 'g', 'b') + ': Saved best model so far to %s.' %(os.path.join(self.save_dir, 'best_model/')))
        if epoch is not None:
            print(cfm('SAVE', 'g', 'b') + ': Saved model for this epoch to %s/%d_*' %(self.save_dir, epoch))

        return
    # ===========================================================================================================================



    def load_checkpoint(self, path, which='latest'):
        """
        Load a model from the saved directory specified. 
        """

        # The argument which specifies which model to load. 
        #   'latest'        :       Load the latest model.
        #   'best'          :       Load the best model from best_model/
        #   <EPOCH>         :       Load the model from the specified epoch.
        if which not in ['latest', 'best'] and not isinstance(which, int):
            print(cfm('HALT', 'r', 'b') + ': Argument \'which\' to load_checkpoint must be either, \'latest\', \'model\', or an epoch number.')
            print(cfm('HALT', 'r', 'b') + ': Argument passed: ', which)
            raise ValueError

        if which == 'latest':
            nets_string                         = os.path.join(path, 'nets.pth')
            optimisers_string                   = os.path.join(path, 'optimisers.pth')
        elif which == 'best':
            nets_string                         = os.path.join(path, 'best_model/', 'nets.pth')
            optimisers_string                   = os.path.join(path, 'best_model/', 'optimisers.pth')
        else:
            nets_string                         = os.path.join(path, '%d_nets.pth' %(which))
            optimisers_string                   = os.path.join(path, '%d_optimisers.pth' %(which))

        if not os.path.exists(nets_string):
            raise ValueError(cfm('HALT', 'r', 'b') + ': Cannot load from %s. Specified path does not exist!' %(nets_string))
        if not os.path.exists(optimisers_string):
            raise ValueError(cfm('HALT', 'r', 'b') + ': Cannot load from %s. Specified path does not exist!' %(optimisers_string))
            

        __saved_models                          = torch.load(nets_string, map_location=self.cpu_map_location)
        __saved_optimisers                      = torch.load(optimisers_string, map_location=self.cpu_map_location)

        for __mname in self.models_dict:
            if __mname in __saved_models:
                __state_dict                    = self.models_dict[__mname].state_dict()
                __pretrained_dict               = {k: v for k, v in __saved_models[__mname].items() if k in __state_dict}
                __state_dict.update(__pretrained_dict)
                self.models_dict[__mname].load_state_dict(__pretrained_dict)

                print('Loading %s ...' %(__mname))
#                self.models_dict[__mname].load_state_dict(__saved_models[__mname])

        for __oname in self.optimisers_dict:
            if __oname in __saved_optimisers:
                __state_dict                    = self.optimisers_dict[__oname].state_dict()
                __pretrained_dict               = {k: v for k, v in __saved_optimisers[__oname].items() if k in __state_dict}
                __state_dict.update(__pretrained_dict)
                self.optimisers_dict[__oname].load_state_dict(__pretrained_dict)
#                self.optimisers_dict[__oname].load_state_dict(__saved_optimisers[__oname])

        for __oname in self.optimisers_dict:
            for pg in self.optimisers_dict[__oname].param_groups:
                pg['lr']                        = self.lr

        print(cfm('LOAD', 'g', 'b') + ': Loaded nets from %s.' %(nets_string))
        print(cfm('LOAD', 'g', 'b') + ': Loaded optimisers from %s.' %(optimisers_string))
        print(cfm('LOAD', 'g', 'b') + ': Optimiser LRs set according to YAML (NOT the saved optimisers).')
        
        return
    # ===========================================================================================================================


    def load_submodel(self, path, which='latest', keys=['cnn'], load_optimisers=False):
        """
        Load only the sub-model from this path. Submodel to load is specified by key.
        """

        if which not in ['latest', 'best'] and not isinstance(which, int):
            raise ValueError(cfm('HALT', 'r', 'b') + ': Argument \'which\' to load_cnn must be either \'latest\', \'best\', or an epoch number\n' + \
                       cfm('HALT', 'r', 'b') + ': Got %s' %(which))

        if which == 'latest':
            nets_string                         = os.path.join(path, 'nets.pth')
            optimisers_string                   = os.path.join(path, 'optimisers.pth')
        elif which == 'best':
            nets_string                         = os.path.join(path, 'best_model/', 'nets.pth')
            optimisers_string                   = os.path.join(path, 'best_model/', 'optimisers.pth')
        else:
            nets_string                         = os.path.join(path, '%d_nets.pth' %(which))
            optimisers_string                   = os.path.join(path, '%d_optimisers.pth' %(which))

        if not os.path.exists(nets_string):
            raise ValueError(cfm('HALT', 'r', 'b') + ': Cannot load from %s. Specified path does not exist!' %(nets_string))
        if not os.path.exists(optimisers_string):
            raise ValueError(cfm('HALT', 'r', 'b') + ': Cannot load from %s. Specified path does not exist!' %(optimisers_string))

        
        __saved_models                          = torch.load(nets_string, map_location=self.cpu_map_location)
        __saved_optimisers                      = torch.load(optimisers_string, map_location=self.cpu_map_location)

        for __k in keys:
            if __k not in __saved_models:
                raise ValueError(cfm('HALT', 'r', 'b') + ': Cannot load %s from the checkpoint at %s because no %s was found in it!' %(__k, nets_string, __k))
       
            __state_dict                        = self.models_dict[__k].state_dict()
            __pretrained_dict                   = {k: v for k, v in __saved_models[__k].items() if k in __state_dict}
            __state_dict.update(__pretrained_dict)
            self.models_dict[__k].load_state_dict(__pretrained_dict)

            print(cfm('LOAD', 'g', 'b') + ': Loaded %s from %s.' %(__k, nets_string))

            if not load_optimisers:
                continue

            if __k in self.optimisers_dict:
                if __k not in __saved_optimisers:
                    raise ValueError(cfm('HALT', 'r', 'b') + ': Cannot load %s optimiser from the checkpoint at %s because no %s was found in it!' %(__k, optimisers_string, __k))
    
                __state_dict                    = self.optimisers_dict[__k].state_dict()
                __pretrained_dict               = {k: v for k, v in __saved_optimisers[__k].items() if k in __state_dict}
                __state_dict.update(__pretrained_dict)
                self.optimisers_dict[__k].load_state_dict(__pretrained_dict)
    
                for pg in self.optimisers_dict[__k].param_groups:
                    pg['lr']                    = self.lr
    
                print(cfm('LOAD', 'g', 'b') + ': Loaded %s optimiser from %s.' %(__k, optimisers_string))
                print(cfm('LOAD', 'g', 'b') + ': Optimiser LRs set according to YAML (NOT the saved optimisers).')

        return
    # ===========================================================================================================================


    def reduce_lr(self):
        """
        Reduce learning rate for all optimisers by a factor determined by self.lr_decay
        """
        for __oname in self.optimisers_dict:
            for pg in self.optimisers_dict[__oname].param_groups:
                pg['lr']                        = pg['lr'] * self.lr_decay
        return 

        
# ===============================================================================================================================
