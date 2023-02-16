import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import pandas as pd
import pyelastix
import pickle
import re

print(os.getcwd() )
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.getcwd(),'elastix-*.*.*-linux/lib')
os.environ['ELASTIX_PATH']= os.path.join(os.getcwd(),'elastix-*.*.*-linux/bin/elastix')

######################################################################################
####This function is for registering one image considering the source_image. 
####You need to install elastix and set above path in order to use this function. 
######################################################################################

def registeration(im1, im2, transform_type): #image1 to be deformed considering image 2
    ##
    # Trasfomation type: RIGID, AFFINE, BSPLINE
    ##
    
    params = pyelastix.get_default_params(type = transform_type)
    params.MaximumNumberOfIterations = 200
    params.FinalGridSpacingInVoxels = 10
    params.NumberOfResolutions = 4
    
    im1_deformed, field = pyelastix.register(im1, im2, params)
    x_max, x_min = im1_deformed.max(), im1_deformed.min()
    im1_deformed = (im1_deformed-x_min) /(x_max - x_min)
#    plt.figure()
#    plt.imshow(im1_deformed)
#    plt.axis('off')
    im1_deformed *= 255
    
    return im1_deformed

######################################################################################
####This function is for reading the excel file and find matching cancer-control patients.
####The output of this function is list of matched ids (cancer_id - control_id) and affected breast-side for cancer and matched control group. 
######################################################################################
def read_xlsx_file():
    """ find matching cancer and control ids"""
    report_file = PATH_TO_INPUT +'new_case_control_year_th_1_age_th_3.xlsx'
    df = pd.read_excel(report_file, sheet_name = 'new_case_control_year_th_1_age_')
    df_dict = df.to_dict()
    matching = {}
    for i,cancer_id in enumerate(df_dict['Cancer'].values()):
        control_id = df_dict['Control'][i]
        matching[cancer_id] = control_id
    
    """ find cancer side outcome for each cancer patient """
    df = pd.read_excel(report_file, sheet_name = 'Cancers')
    df_dict = df.to_dict()
    cancer_side = {}
    for i,cancer_id in enumerate(df_dict['StudyID'].values()):
        left = df_dict['L current outcome'][i]
        if left in ['7','9','15','19']:
            cancer_side[cancer_id]=0  #for left
        else:
            cancer_side[cancer_id]=2  #for right
    return matching, cancer_side

######################################################################################
####This function is for creating list of patients with at least NO_TIME_STEPS-1 priors and defined the final number of patients
####The output is list of lists
######################################################################################
def create_matching_patients(NO_TIME_STEPS):

    cancer_dir = os.path.join(PATH_TO_INPUT,'Patients1/Cancer')
    control_dir = os.path.join(PATH_TO_INPUT, 'Patients1/Control')

    cancer_dirs = [ids for ids in sorted(os.listdir(cancer_dir))]
    control_dirs = [ids for ids in sorted(os.listdir(control_dir))]

    cncr_cntrl_match,cncr_side = read_xlsx_file()

    matching_patients = [(str(patient_id),str(cncr_cntrl_match[patient_id]))
                     for patient_id in cancer_dirs 
                     if (len(os.listdir(os.path.join(cancer_dir ,patient_id))) >= NO_TIME_STEPS) 
                     and (len(os.listdir(os.path.join(control_dir ,cncr_cntrl_match[patient_id]))) >= NO_TIME_STEPS)]

    return matching_patients

######################################################################################
####This function is for reading image paths and do preprocessing
######################################################################################
def read_patient_file(file_path, height,width, side, source_image):
    
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if height is not None and width is not None: 
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
    if side == 'right' and source_image is not None: 
        image = cv2.flip(image, 1)
        image = registeration(image, source_image, 'AFFINE')
    elif side == 'left' and source_image is not None: 
        image = registeration(image, source_image, 'AFFINE')
    
    return image
    

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key,reverse=False) 

class Dataset:
    
    def __init__(
            self,
            input_dir,
            matching_patients,
            cancer_dir,
            control_dir,
            augmentation = None,
            preprocessing = None,
            
            ):
        """ make a list of image directions  """
        
        self.matching_fps = (' '.join(w for w in re.split(r"\W", create_matching_patients(NO_TIME_STEPS = 5)) if w)).split(' ') # 5 for 4 priors, 4 for 3 priors ,...
        
        self.patient_ids        = []
        self.patient_images     = []
        self.patient_labels     = []
        print(len(self.matching_fps))
        
        for w in range(0, len(self.matching_fps), 2):
            cancer_id, control_id = self.matching_fps[w], self.matching_fps[w+1]
            cancer_path = os.path.join(cancer_dir ,cancer_id)
            control_path = os.path.join(control_dir, control_id)
            
            cancer_priors= sorted_alphanumeric(os.listdir(cancer_path))
            
            control_priors = sorted_alphanumeric(os.listdir(control_path))
            
            prior_images_left = []
            prior_images_right = []
           
            for prior in cancer_priors:
                two_images_left = []
                two_images_right = []
                images = sorted_alphanumeric(os.listdir(os.path.join(cancer_path ,prior)))
               
                if(len(images) < 4):
                    print("{} of this {} has less than 4 images and need to be checked".format(prior,cancer_path))
                cc_left , mlo_left,cc_right,mlo_right = images[0],images[1],images[2],images[3]
                two_images_left.append(os.path.join(cancer_path,prior,cc_left))
                two_images_left.append(os.path.join(cancer_path,prior,mlo_left))
                prior_images_left.append(two_images_left) 
                
                two_images_right.append(os.path.join(cancer_path,prior,cc_right))
                two_images_right.append(os.path.join(cancer_path,prior,mlo_right))
                prior_images_right.append(two_images_right) 
            
            self.patient_ids.append(cancer_id)
            self.patient_images.append([prior_images_left,prior_images_right])
            self.patient_labels.append(1.0)    
            
            prior_images_left = []
            prior_images_right = []
            for prior in control_priors:
                two_images_left = []
                two_images_right = []
                images = sorted_alphanumeric(os.listdir(os.path.join(control_path ,prior)))
                if(len(images) < 4):
                    print("{} of this {} has less than 4 images and need to be checked".format(prior,control_path))
                    
                cc_left , mlo_left,cc_right,mlo_right = images[0],images[1],images[2],images[3]
                two_images_left.append(os.path.join(control_path,prior,cc_left))
                two_images_left.append(os.path.join(control_path,prior,mlo_left))
                prior_images_left.append(two_images_left) 
                
                two_images_right.append(os.path.join(control_path,prior,cc_right))
                two_images_right.append(os.path.join(control_path,prior,mlo_right))
                prior_images_right.append(two_images_right)  

            self.patient_ids.append(control_id)
            self.patient_images.append([prior_images_left,prior_images_right])
            self.patient_labels.append(0.0)
        
        self.patient_labels = to_categorical(self.patient_labels,2)
        """ set the class values and assign a augmentation and preprocessing method"""
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __getitem__(self,i):
        
        if MODEL_NAME == '_all_TIMES':
          
            ####prior_1
            prior_cc_1_left = read_patient_file(self.patient_images[i][0][1][0], height=None,width=None, side='left', source_image=None)
            """ need this line to resizing all the steps to the fixed height and width"""
            w, h,_ = prior_cc_1_left.shape 
            
            prior_cc_1_right = read_patient_file(self.patient_images[i][1][1][0], height=h,width=w, side='right', source_image=prior_cc_1_left)
            prior_mlo_1_left = read_patient_file(self.patient_images[i][0][1][1], height=None,width=None, side='left', source_image=None)
            prior_mlo_1_right = read_patient_file(self.patient_images[i][1][1][1], height=h,width=w, side='right', source_image=prior_mlo_1_left)
            
            ####prior_2
            prior_cc_2_left = read_patient_file(self.patient_images[i][0][2][0], height=h,width=w, side='left', source_image=prior_cc_1_left)
            prior_cc_2_right = read_patient_file(self.patient_images[i][1][2][0], height=h,width=w, side='right', source_image=prior_cc_1_left)
            prior_mlo_2_left = read_patient_file(self.patient_images[i][0][2][1], height=h,width=w, side='left', source_image=prior_mlo_1_left)
            prior_mlo_2_right =read_patient_file(self.patient_images[i][1][2][1], height=h,width=w, side='right', source_image=prior_mlo_1_left)
            
            ####prior_3
            prior_cc_3_left = read_patient_file(self.patient_images[i][0][3][0], height=h,width=w, side='left', source_image=prior_cc_1_left)
            prior_cc_3_right = read_patient_file(self.patient_images[i][1][3][0], height=h,width=w, side='right', source_image=prior_cc_1_left)
            prior_mlo_3_left = read_patient_file(self.patient_images[i][0][3][1], height=h,width=w, side='left', source_image=prior_mlo_1_left)
            prior_mlo_3_right =read_patient_file(self.patient_images[i][1][3][1], height=h,width=w, side='right', source_image=prior_mlo_1_left)
            
            ####prior_4
            prior_cc_4_left = read_patient_file(self.patient_images[i][0][4][0], height=h,width=w, side='left', source_image=prior_cc_1_left)
            prior_cc_4_right = read_patient_file(self.patient_images[i][1][4][0], height=h,width=w, side='right', source_image=prior_cc_1_left)
            prior_mlo_4_left = read_patient_file(self.patient_images[i][0][4][1], height=h,width=w, side='left', source_image=prior_mlo_1_left)
            prior_mlo_4_right =read_patient_file(self.patient_images[i][1][4][1], height=h,width=w, side='right', source_image=prior_mlo_1_left)
            
        
        final_label = self.patient_labels[i]
        
        data1 = {"image": prior_cc_1_left, "MLO_1_left": prior_mlo_1_left, "CC_1_right": prior_cc_1_right, "MLO_1_right": prior_mlo_1_right,
                 "CC_2_left": prior_cc_2_left, "MLO_2_left": prior_mlo_2_left,"CC_2_right": prior_cc_2_right, "MLO_2_right": prior_mlo_2_right,
                 "CC_3_left": prior_cc_3_left, "MLO_3_left": prior_mlo_3_left,"CC_3_right": prior_cc_3_right, "MLO_3_right": prior_mlo_3_right,
                 "CC_4_left": prior_cc_4_left, "MLO_4_left": prior_mlo_4_left,"CC_4_right": prior_cc_4_right, "MLO_4_right": prior_mlo_4_right}
                
        """ apply augmentation """
        if self.augmentation:
            sample = self.augmentation(**data1)
            prior_cc_1_left , prior_mlo_1_left, prior_cc_2_left , prior_mlo_2_left, prior_cc_3_left , prior_mlo_3_left, prior_cc_4_left , prior_mlo_4_left = sample['image'], sample['MLO_1_left'],sample['CC_2_left'], sample['MLO_2_left'],sample['CC_3_left'], sample['MLO_3_left'],sample['CC_4_left'], sample['MLO_4_left']
            prior_cc_1_right , prior_mlo_1_right, prior_cc_2_right , prior_mlo_2_right, prior_cc_3_right , prior_mlo_3_right, prior_cc_4_right , prior_mlo_4_right = sample['CC_1_right'], sample['MLO_1_right'],sample['CC_2_right'], sample['MLO_2_right'],sample['CC_3_right'], sample['MLO_3_right'],sample['CC_4_right'], sample['MLO_4_right']
         
        data2 = {"image": prior_cc_1_left, "MLO_1_left": prior_mlo_1_left, "CC_1_right": prior_cc_1_right, "MLO_1_right": prior_mlo_1_right,
                 "CC_2_left": prior_cc_2_left, "MLO_2_left": prior_mlo_2_left,"CC_2_right": prior_cc_2_right, "MLO_2_right": prior_mlo_2_right,
                 "CC_3_left": prior_cc_3_left, "MLO_3_left": prior_mlo_3_left,"CC_3_right": prior_cc_3_right, "MLO_3_right": prior_mlo_3_right,
                 "CC_4_left": prior_cc_4_left, "MLO_4_left": prior_mlo_4_left,"CC_4_right": prior_cc_4_right, "MLO_4_right": prior_mlo_4_right}
         
        """ apply preprocessing """
        if self.preprocessing:
            sample = self.preprocessing(**data2)
            prior_cc_1_left , prior_mlo_1_left, prior_cc_2_left , prior_mlo_2_left, prior_cc_3_left , prior_mlo_3_left, prior_cc_4_left , prior_mlo_4_left = sample['image'], sample['MLO_1_left'],sample['CC_2_left'], sample['MLO_2_left'],sample['CC_3_left'], sample['MLO_3_left'],sample['CC_4_left'], sample['MLO_4_left']
            prior_cc_1_right , prior_mlo_1_right, prior_cc_2_right , prior_mlo_2_right, prior_cc_3_right , prior_mlo_3_right, prior_cc_4_right , prior_mlo_4_right = sample['CC_1_right'], sample['MLO_1_right'],sample['CC_2_right'], sample['MLO_2_right'],sample['CC_3_right'], sample['MLO_3_right'],sample['CC_4_right'], sample['MLO_4_right']
         
        return prior_cc_1_left , prior_mlo_1_left, prior_cc_2_left , prior_mlo_2_left, prior_cc_3_left , prior_mlo_3_left, prior_cc_4_left , prior_mlo_4_left,prior_cc_1_right , prior_mlo_1_right, prior_cc_2_right , prior_mlo_2_right, prior_cc_3_right , prior_mlo_3_right, prior_cc_4_right , prior_mlo_4_right, final_label,self.patient_ids[i] 
    
    def __len__(self):
        return len(self.patient_ids)