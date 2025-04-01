from main import *

if __name__ == '__main__':
    logging.info(f"Starting")
    # Call the function to set the seed
    set_seed(42)

    mapping = {
        '10' : 8.5,
        '5' : 6.5,
        '0' : 2
    }

    dataset_name = "pca_transformed_test_validation_1.csv"
    dataset_dir  = "/home/mojtaba.kananisarche/Zh/UTA/feature_vectors/mobilenetv2/pca_transformed/"
    dataset_path = os.path.join(dataset_dir, dataset_name)
    df = pd.read_csv(dataset_path)
    df['target'] = df['target'].astype(str)
    df['target'] = df['filename'].apply(lambda x:mapping.get(x.split('_')[2]))



    m = Model(units = [100, 50], 
            dropout_ratio = 0.1, 
            kernel_regularizer_l2 = 0, 
            kernel_regularizer_l1 = 0, 
            batch_size = 16, 
            num_classes = 3, 
            learning_rate = 1e-3, 
            weight_decay = 1e-4, 
            beta_1 = 0.9, 
            beta_2 = 0.999, 
            epsilon = 1e-7,
            epochs = 1000, 
            patience = 100,
            max_slices=3000,
            mapping = mapping,
            dataset_name = dataset_name)

    m.test_model(df)