img_size = 224
num_classes = 200
prototype_shape = (num_classes*10, 64, 1, 1)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'


data_path = '/mnt/c/copy/data/CUB_200_2011_full/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train'


data_path_ood1 = '/mnt/c/copy/data/Cars_full/'
test_dir_ood1 = data_path_ood1 + 'traintest/'


data_path_ood2 = '/mnt/c/copy/data/Pets_full/'
test_dir_ood2 = data_path_ood2 + 'traintest/'


train_batch_size = 80  # 80
test_batch_size = 80
train_push_batch_size = 80


joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}   # 3e-3
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4


coefs = {
    'crs_ent': 1,
    'mine': 0.2,
    'aux': 0.5,
}


num_train_epochs = 120  # 180
num_warm_epochs = 0

mine_start = 40        # 10 (R50)
updateGMM_start = 35   # 10 (R50)

push_start = 100   # 100
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]