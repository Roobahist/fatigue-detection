from main import *

if __name__ == "__main__":
    logging.info(f"Starting")
    # Call the function to set the seed
    set_seed(42)
    
    pca_transformed_path = "/home/mojtaba.kananisarche/Zh/UTA/feature_vectors/mobilenetv2/pca_transformed"

    mapping = {
        '10' : 10,
        '5'  : 5,
        '0'  : 0
    }
    
#     df['fold']       = df['filename'].apply(lambda x: int(x.split('_')[0]))
#     df['subject_id'] = df['filename'].apply(lambda x: int(x.split('_')[1]))
#     df['target']     = df['filename'].apply(lambda x:mapping.get(x.split('_')[2]))
    
    m = Model(units = [100, 100], 
              dropout_ratio = 0.1, 
              kernel_regularizer_l2 = 0, 
              kernel_regularizer_l1 = 0, 
              batch_size = 8, 
              num_classes = 3, 
              learning_rate = 1e-5, 
              weight_decay = 1e-4, 
              beta_1 = 0.999, 
              beta_2 = 0.999, 
              epsilon = 1e-7,
              epochs = 1000, 
              patience = 300,
              max_slices=200,
              mapping = mapping,
              pca_transformed_path = pca_transformed_path)

    m.test_model()
