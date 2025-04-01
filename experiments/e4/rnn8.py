from main import *

if __name__ == "__main__":
    logging.info(f"Starting")
    # Call the function to set the seed
    set_seed(42)
    
    dataset_name = "Blendshapes_Summary_50_50_0.7_45_3.csv"
    df = pd.read_csv(f"/home/mojtaba.kananisarche/Zh/UTA/Blendshapes/Summary/{dataset_name}", index_col=0)

    mapping = {
        '10' : 8.5,
        '5' : 6.5,
        '0' : 2
    }
    df['fold'] = df['filename'].apply(lambda x: int(x.split('_')[0]))
    df['subject_id'] = df['filename'].apply(lambda x: int(x.split('_')[1]))
    df['target'] = df['filename'].apply(lambda x:mapping.get(x.split('_')[2]))
    
    
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
            epochs = 1000, 
            patience = 100,
            max_slices=100,
            dataset_name = dataset_name)

    m.test_model(df)
