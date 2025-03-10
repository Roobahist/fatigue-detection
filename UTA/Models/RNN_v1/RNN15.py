from RNN_Main import *

if __name__ == "__main__":
    logging.info(f"Starting")
    # Call the function to set the seed
    set_seed(42)
    
    dataset_name = "Blendshapes_Summary_50_50_0.5_45_3.csv"
    
    r = pd.read_csv(f"/home/mojtaba.kananisarche/Zh/UTA/Blendshapes/Summary/{dataset_name}", index_col=0)
    r['target'] = r['filename'].apply(lambda x: replace_target(x.split('_')[2], {'10': 8.5, '0': 2.0, '5': 6.5}))
    r['fold'] = r['filename'].apply(lambda x: int(x.split('_')[0]))
    r['subject_id'] = r['filename'].apply(lambda x: int(x.split('_')[1]))
    
    
    m = Model(units = [400, 400, 400], 
          dropout_ratio = 0.3, 
          kernel_regularizer_l2 = 0, 
          kernel_regularizer_l1 = 0, 
          batch_size = 32, 
          num_classes = 3, 
          learning_rate = 1e-3, 
          weight_decay = 1e-4, 
          beta_1 = 0.9, 
          beta_2 = 0.999, 
          epsilon = 1e-7, 
          cv_on = 'subject_id', 
          epochs = 1000, 
          patience = 100,
          max_slices=100,
          dataset_name = dataset_name)

    m.test_model(r)

    
