import os
from shutil import copyfile

def pytorch_prepare(data_dir, dataset_name):
    dataset_dir = os.path.join(data_dir, dataset_name)

    if not os.path.isdir(dataset_dir):
        print('please change the download_path')

    pytorch_path = os.path.join(dataset_dir , 'pytorch')

    if not os.path.isdir(pytorch_path):
        os.mkdir(pytorch_path)
        #-----------------------------------------
        #query
        print('generatring ' + dataset_name + ' query images.')
        query_dir = os.path.join(dataset_dir , 'query')
        query_save_dir = os.path.join(dataset_dir , 'pytorch', 'query')
        if not os.path.isdir(query_save_dir):
            os.mkdir(query_save_dir)

        for root, dirs, files in os.walk(query_dir, topdown=True):
            for name in files:
                if not name[-3:]=='jpg':
                    continue
                ID  = name.split('_')
                src_dir = os.path.join(query_dir , name)
                dst_dir = os.path.join(query_save_dir, ID[0])
                if not os.path.isdir(dst_dir):
                    os.mkdir(dst_dir)
                copyfile(src_dir, os.path.join(dst_dir , name))
        #-----------------------------------------
        #gallery
        print('generatring '+dataset_name+' gallery images.')
        gallery_dir = os.path.join(dataset_dir , 'bounding_box_test')
        gallery_save_dir = os.path.join(dataset_dir , 'pytorch' , 'gallery')
        if not os.path.isdir(gallery_save_dir):
            os.mkdir(gallery_save_dir)

        for root, dirs, files in os.walk(gallery_dir, topdown=True):
            for name in files:
                if not name[-3:]=='jpg':
                    continue
                ID  = name.split('_')
                src_dir = os.path.join(gallery_dir, name)
                dst_dir = os.path.join(gallery_save_dir, ID[0])
                if not os.path.isdir(dst_dir):
                    os.mkdir(dst_dir)
                copyfile(src_dir, os.path.join(dst_dir,name))
        #---------------------------------------
        #train_all
        print('generatring '+dataset_name + ' all training images.')
        train_dir = os.path.join( dataset_dir , 'bounding_box_train')
        train_save_all_dir = os.path.join( dataset_dir , 'pytorch', 'train_all')
        if not os.path.isdir(train_save_all_dir):
            os.mkdir(train_save_all_dir)

        for root, dirs, files in os.walk(train_dir, topdown=True):
            for name in files:
                if not name[-3:]=='jpg':
                    continue
                ID  = name.split('_')
                src_dir = os.path.join(train_dir , name)
                dst_dir = os.path.join(train_save_all_dir, ID[0])
                if not os.path.isdir(dst_dir):
                    os.mkdir(dst_dir)
                copyfile(src_dir, os.path.join(dst_dir, name))
                
        #---------------------------------------
        #train_val
        print('generatring '+ dataset_name+' training and validation images.')
        train_save_dir = os.path.join(dataset_dir, 'pytorch', 'train')
        val_save_dir = os.path.join(dataset_dir , 'pytorch' , 'val')
        if not os.path.isdir(train_save_dir):
            os.mkdir(train_save_dir)
            os.mkdir(val_save_dir)

        for root, dirs, files in os.walk(train_dir, topdown=True):
            for name in files:
                if not name[-3:]=='jpg':
                    continue
                ID  = name.split('_')
                src_dir = os.path.join(train_dir , name)
                dst_dir = os.path.join(train_save_dir , ID[0])
                if not os.path.isdir(dst_dir):
                    os.mkdir(dst_dir)
                    dst_dir = os.path.join(val_save_dir, ID[0])  #first image is used as val image
                    os.mkdir(dst_dir)
                copyfile(src_dir, os.path.join(dst_dir , name))
        print('Finished ' + dataset_name)
    else:
        print(dataset_name + ' pytorch directory exists!')

def pytorch_prepare_all(data_dir):
    pytorch_prepare('Market1501', data_dir)
    pytorch_prepare('DukeMTMC', data_dir)
