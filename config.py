from easydict import EasyDict as config
cfg = config()

# class labels
cfg.class_label = {'background':0, 'stem':1, 'callus':2,  'shoot':3}

# class colors
cfg.class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]

# base directory where the label and images are placed                    
cfg.dirname='./In_vitro_set/'

# base directory of the output folder
cfg.base_output_path='./Annotation'

# subdirectories for colored labels, uncolored labels and merged label images with rgb (for visualization only)
cfg.lbl_path='label'

# output list name
cfg.output_list_name = 'test.txt'

