class Dataloader(tf.keras.utils.Sequence):
    
    def __init__(
            self,
            dataset,
            batch_size = 1,
            shuffle = False
            ):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        
        self.on_epoch_end()
        
    def __getitem__(self, i):
        
        start = i * self.batch_size
        stop = (i+1) * self.batch_size
        data_cc_1_left = []
        data_mlo_1_left = []
        data_cc_2_left = []
        data_mlo_2_left = []
        data_cc_3_left = []
        data_mlo_3_left = []
        data_cc_4_left = []
        data_mlo_4_left = []
        
        data_cc_1_right = []
        data_mlo_1_right = []
        data_cc_2_right = []
        data_mlo_2_right = []
        data_cc_3_right = []
        data_mlo_3_right = []
        data_cc_4_right = []
        data_mlo_4_right = []
        
        labels = []
        for j in range(start, stop):
            prior_cc_1_left , prior_mlo_1_left, prior_cc_2_left , 
            prior_mlo_2_left, prior_cc_3_left , prior_mlo_3_left, 
            prior_cc_4_left , prior_mlo_4_left,prior_cc_1_right , 
            prior_mlo_1_right, prior_cc_2_right , prior_mlo_2_right, 
            prior_cc_3_right , prior_mlo_3_right, prior_cc_4_right , 
            prior_mlo_4_right,label,_ = self.dataset[j]
#           
            data_cc_1_left.append(prior_cc_1_left)
            data_mlo_1_left.append(prior_mlo_1_left)
            data_cc_2_left.append(prior_cc_2_left)
            data_mlo_2_left.append(prior_mlo_2_left)
            data_cc_3_left.append(prior_cc_3_left)
            data_mlo_3_left.append(prior_mlo_3_left)
            data_cc_4_left.append(prior_cc_4_left)
            data_mlo_4_left.append(prior_mlo_4_left)
            
            data_cc_1_right.append(prior_cc_1_right)
            data_mlo_1_right.append(prior_mlo_1_right)
            data_cc_2_right.append(prior_cc_2_right)
            data_mlo_2_right.append(prior_mlo_2_right)
            data_cc_3_right.append(prior_cc_3_right)
            data_mlo_3_right.append(prior_mlo_3_right)
            data_cc_4_right.append(prior_cc_4_right)
            data_mlo_4_right.append(prior_mlo_4_right)
            
            labels.append(label)
#         batch = [np.stack(samples, axis = 0) for samples in zip(*data)]
        return [np.array(data_cc_1_left),np.array(data_cc_1_right)
                ,np.array(data_cc_2_left),np.array(data_cc_2_right)
                ,np.array(data_cc_3_left),np.array(data_cc_3_right)
                ,np.array(data_cc_4_left),np.array(data_cc_4_right)
                ,np.array(data_mlo_1_left),np.array(data_mlo_1_right)
                ,np.array(data_mlo_2_left),np.array(data_mlo_2_right)
                ,np.array(data_mlo_3_left),np.array(data_mlo_3_right)
                ,np.array(data_mlo_4_left),np.array(data_mlo_4_right)],np.array(labels)
#         return batch            
    
    def __len__(self):
        """  the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)  