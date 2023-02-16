import tensorflow as tf
from Model import *


    
def run(fold,train_dataloader,valid_dataloader,EPOCHS,metrics):
if not os.path.exists(PATH_TO_SAVE_MODEL+'/'+MODEL_NAME+'/'+MODEL_DESC+'/'):
    os.makedirs(PATH_TO_SAVE_MODEL+'/'+MODEL_NAME+'/'+MODEL_DESC+'/')

model = create_model(metrics,(224,224,3),32, 512)
print(model.summary())
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(PATH_TO_SAVE_MODEL+'/'+MODEL_NAME+'/'+MODEL_DESC+'/fold_'+str(fold), save_weights_only=True,monitor='val_auc', save_best_only=False,save_freq="epoch", mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(),
    ]

history = model.fit_generator(
    train_dataloader,
    steps_per_epoch = len(train_dataloader),
    epochs = EPOCHS,
    callbacks = callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
    )

return history