import albumentations as A

def round_clip_0_1(x, **kwargs):
    return x.round

def training_augmentation():
    train_transform = [
        A.PadIfNeeded(min_height = 224, min_width = 224, always_apply= True, border_mode=0),
        A.Resize(height=224, width=224,always_apply=True, interpolation=1, p=1),

    ]

    return A.Compose(train_transform,
        additional_targets={ "MLO_1_left":"image","CC_1_right":"image","MLO_1_right":"image",
                            "CC_2_left" : "image", "MLO_2_left":"image","CC_2_right" : "image", "MLO_2_right":"image",
                            "CC_3_left" : "image", "MLO_3_left":"image","CC_3_right" : "image", "MLO_3_right":"image",
                            "CC_4_left" : "image", "MLO_4_left":"image","CC_4_right" : "image", "MLO_4_right":"image",})

def validation_augmentation():
    test_transform = [
        A.PadIfNeeded(min_height = 256, min_width = 256, always_apply= True, border_mode=0),
        A.Resize(height=256, width=256, interpolation=1, always_apply=True, p=1)
        ]
    return A.Compose(test_transform,
            additional_targets={ "MLO_1_left":"image","CC_1_right":"image","MLO_1_right":"image",
                                "CC_2_left" : "image", "MLO_2_left":"image","CC_2_right" : "image", "MLO_2_right":"image",
                                "CC_3_left" : "image", "MLO_3_left":"image","CC_3_right" : "image", "MLO_3_right":"image",
                                "CC_4_left" : "image", "MLO_4_left":"image","CC_4_right" : "image", "MLO_4_right":"image",})

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image = preprocessing_fn),
        ]
    return A.Compose(_transform,
            additional_targets={ "MLO_1_left":"image","CC_1_right":"image","MLO_1_right":"image",
                                "CC_2_left" : "image", "MLO_2_left":"image","CC_2_right" : "image", "MLO_2_right":"image",
                                "CC_3_left" : "image", "MLO_3_left":"image","CC_3_right" : "image", "MLO_3_right":"image",
                                "CC_4_left" : "image", "MLO_4_left":"image","CC_4_right" : "image", "MLO_4_right":"image",})
