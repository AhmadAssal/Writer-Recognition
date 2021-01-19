#-------------------------imports-----------------------------
import sys
import numpy as np
import os
import cv2 as cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
from sklearn.svm import SVR
from skimage import feature
from skimage.exposure import histogram
from tqdm import tqdm
from sklearn.svm import LinearSVC
import random
from datetime import datetime
import shutil
import time
import glob,fnmatch

#----------------------Utilities------------------------------
def show(img, factor=1,name="image"):
    """ 
    show an image until the escape key is pressed
    :param factor: scale factor (default 1, half size)
    """
    if factor != 1.0:
        img = cv2.resize(img, (0,0), fx=factor, fy=factor) 

    cv2.imshow(name,img)
    while(1):
        k = cv2.waitKey(0)
        if k==27:    # Esc key to stop
            break
    cv2.destroyAllWindows()


def read_image(input_path):
    img = cv2.imread(input_path,0)
    return img




def crop_img(img):
    mask_inv = img
    ver_sum = np.sum(mask_inv,axis=1)
    v_start = 0
    v_end = 0
    for i in range(len(ver_sum)):
        if(ver_sum[i] > 0 and v_start ==0):
            v_start = i
        if(ver_sum[i] == 0 and v_start != 0):
            v_end = i
            break
    if(v_end == 0):
        v_end = len(ver_sum) - 1
    
    hor_sum = np.sum(mask_inv,axis=0)
    h_start = 0
    h_end = 0
    for i in range(len(hor_sum)):
        if(hor_sum[i] > 0 and h_start ==0):
            h_start = i
        if(hor_sum[i] == 0 and h_start != 0):
            h_end = i
            break
    if(h_end == 0):
        h_end = len(hor_sum) - 1

    return img[v_start:v_end,h_start:h_end]


#---------------Preprocessing------------------------


def preprocess_img(img,show_steps=1,show_size=0.2):
    img = cv2.GaussianBlur(img,(11,11),0)
    thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img[img >= thresh[0]] = 255
    img[img <= thresh[0]] = 0
    img = cv2.bitwise_not(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if show_steps == 1:        
        show(img,show_size,"Preprocessed Image")
    return img

def remove_top(binary_img,gray_img,show_steps=1,show_size=0.2):

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    y_list = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > binary_img.shape[1] / 2 and w < binary_img.shape[1] * 5 / 6:
            y_list.append(y)
    y_list = np.sort(y_list)
    y1=y_list[-2]
    y2=y_list[-1]
    if show_steps == 1:        
        show(gray_img[y1+20:y2-20,:],show_size,"Cropped Image")
    return binary_img[y1+20:y2-20,:],gray_img[y1+20:y2-20,:]


def sort_lists(list1,list2):
    list1 = np.asarray(list1)
    sorter = np.argsort(list1)
    list3 = []
    for i in range(len(sorter)):
        list3.append(list2[sorter[i]])
    return list3


def merge_boxes(boxes,max_h):
    i = 0
    while i < len(boxes)-1:
        j = i+1
        while j < len(boxes):
            x1,y1,w1,h1 = (boxes[i])[0],(boxes[i])[1],(boxes[i])[2],(boxes[i])[3]
            x2,y2,w2,h2 = (boxes[j])[0],(boxes[j])[1],(boxes[j])[2],(boxes[j])[3]
            if x2 < x1+w1:
                boxes[i] = [x1,min(y1,y2),x2+w2-x1,max(y1+h1,y2+h2)-min(y1,y2)]
                del boxes[j]
            else:
                break
        i += 1
    return boxes


def split_block(binary_img,gray_img,show_steps,show_size):

    original_binary = binary_img.copy()
    original_gray = gray_img.copy()       
    if show_steps == 1:        
        show(binary_img,show_size,"Block Separation")
    
    cnts,_ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    avg_h = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        avg_h += h
    avg_h /= len(cnts)
    avg_h *= 1.75
    b_sents = []
    g_sents = []
    
    hor_sum = np.sum(binary_img, axis=1)
    max_line = np.amax(hor_sum)
    h_c = 0
    for i in range(len(hor_sum)):
        h_c += 1
        if(h_c > avg_h and hor_sum[i] < 0.4*max_line):
            if( (i-avg_h) >avg_h and np.sum(original_binary[int(i-avg_h):i,:]) > 0.05*255*original_binary[int(i-avg_h):i,:].shape[0]*original_binary[int(i-avg_h):i,:].shape[1]):
                b_sents.append(original_binary[int(i-avg_h):i,:])
                g_sents.append(original_gray[int(i-avg_h):i,:])
                if show_steps == 1:        
                    show(original_gray[int(i-avg_h):i,:],1,"Separated Block Number : " + str(len(b_sents)))
                h_c = 0
            
    
    return b_sents,g_sents



def split_block(binary_img,gray_img,show_steps,show_size):

    original_binary = binary_img.copy()
    original_gray = gray_img.copy()       
    if show_steps == 1:        
        show(binary_img,show_size,"Block Separation")
    
    cnts,_ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    avg_h = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        avg_h += h
    avg_h /= len(cnts)
    avg_h *= 1.75
    b_sents = []
    g_sents = []
    
    hor_sum = np.sum(binary_img, axis=1)
    max_line = np.amax(hor_sum)
    h_c = 0
    for i in range(len(hor_sum)):
        h_c += 1
        if(h_c > avg_h and hor_sum[i] < 0.4*max_line):
            if( (i-avg_h) >avg_h and np.sum(original_binary[int(i-avg_h):i,:]) > 0.05*255*original_binary[int(i-avg_h):i,:].shape[0]*original_binary[int(i-avg_h):i,:].shape[1]):
                b_sents.append(original_binary[int(i-avg_h):i,:])
                g_sents.append(original_gray[int(i-avg_h):i,:])
                if show_steps == 1:        
                    show(original_gray[int(i-avg_h):i,:],1,"Separated Block Number : " + str(len(b_sents)))
                h_c = 0
            
    
    return b_sents,g_sents


def get_horizontal_merge(binary_sentence,gray_sentence,show_steps=1,show_size=0.2):
    original = (gray_sentence*(binary_sentence/255)).astype(np.uint8)
    if show_steps == 1:        
        show(original,show_size*2,"Sentence Before Horizontal Merge")
    # Find contours
    cnts,_ = cv2.findContours(binary_sentence, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate thorugh contours and filter for ROI
    order_list = []
    boxes = []
    images = []
    area = 0 
    avg_height = 0
    for c in cnts:
        area += cv2.contourArea(c)
    area /= len(cnts)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if cv2.contourArea(c) > area/8 :
            boxes.append([x,y,w,h])
            order_list.append(x)
        else:
            original[y:y+h,x:x+w] = 0
    boxes = sort_lists(order_list,boxes)
    boxes = merge_boxes(boxes,original.shape[0])
    for box in boxes:
        x,y,w,h = box[0],box[1],box[2],box[3]                    
        ROI = np.zeros((original.shape[0],w))
        avg_height += h
        if int(original.shape[0]/2-h/2) > 0 :
            ROI[int(original.shape[0]/2-h/2):int(original.shape[0]/2-h/2)+h,:] = original[y:y+h,x:x+w]
        else:           
            ROI[0:h,:] = original[y:y+h,x:x+w]
        images.append(ROI)
    avg_height /= len(boxes)
    hori_merged = np.zeros((original.shape[0],original.shape[1]))
    current = 0
    for image in images:
        hori_merged[:,current:current+image.shape[1]] += image
        current = current+image.shape[1]
    hori_merged = crop_img(hori_merged)
    hori_merged = hori_merged.astype(np.uint8)
    if show_steps == 1:        
        show(hori_merged,show_size*5,"Sentence After Horizontal Merge")

    return hori_merged,avg_height

def get_sentences(binary_img,gray_img,show_steps=1,show_size=0.2):
    
    original_binary = binary_img.copy()
    original_gray = gray_img.copy()    
    canny = cv2.Canny(binary_img, 200, 400)
    kernel = np.ones((1,7),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=40)
    
    if show_steps == 1:        
        show(dilate,show_size,"Dialted for Sentence Extraction")
    
    # Find contours
    cnts,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate thorugh contours and filter for ROI
    image_number = 0
    order_list = []
    binary_images = []
    gray_images = []
    area = 0 
    for c in cnts:
        area += cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
    area /= (len(cnts))
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if cv2.contourArea(c) > area/2:

            if (w < binary_img.shape[0]/5)  or (h > binary_img.shape[0]/6 and w < binary_img.shape[0]/5):
                continue
            cv2.rectangle(binary_img, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI_binary = original_binary[y:y+h, x:x+w]
            ROI_gray = original_gray[y:y+h, x:x+w]

            if h > binary_img.shape[0]/7:
                b_sents,g_sents = split_block(ROI_binary,ROI_gray,show_steps,show_size)
                for i in range(len(b_sents)):
                    order_list.append(y+ (i/len(b_sents))*h )
                    binary_images.append(b_sents[i])
                    gray_images.append(g_sents[i])
            else:
                binary_images.append(ROI_binary)
                gray_images.append(ROI_gray)
                order_list.append(y)

    sentences_binary = sort_lists(order_list,binary_images)
    sentences_gray = sort_lists(order_list,gray_images)
    return sentences_binary,sentences_gray

def rearrange_image(binary_img,gray_img,show_steps=1,show_size=0.2):
    
    copy = np.zeros((binary_img.shape[0], binary_img.shape[1]))
    binary_sentences,gray_sentences = get_sentences(binary_img,gray_img,show_steps,show_size)
    contours, hierarchy = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    currentY = 150
    for i in range(len(binary_sentences)):
        
        sentence,avg_height = get_horizontal_merge(binary_sentences[i],gray_sentences[i],show_steps,show_size)
        ROI = copy[currentY:currentY+sentence.shape[0],0:sentence.shape[1]]
#         ROI += sentence
        ROI[ROI == 0] = sentence[ROI == 0]
        copy[currentY:currentY+sentence.shape[0],0:sentence.shape[1]] += ROI
        currentY += int(avg_height/2)
    copy = crop_img(copy)
    copy = copy.astype(np.uint8)
    if show_steps == 1:        
        show(copy,show_size*2,"Vertical Merge")
    return copy


def divide_image(image,show_steps=1,show_size=0.2):
    factor = 3
    height, width = image.shape
    img_arr = []
    w_3 =int(width/3)
    h_3= int(height/3)

    for i in range(9):
        rand_row = int((random.random()*image.shape[0]) %(image.shape[0] - 128))
        ran_column = int((random.random()*image.shape[1]) %(image.shape[1] - 256))
#         img_arr.append(image[rand_row:rand_row+128 , ran_column:ran_column+256])
        img_arr.append( image[int(image.shape[0]/2 - 64):int(image.shape[0]/2 + 64) , ran_column:ran_column+256])

    if show_steps == 1:
        for img in img_arr:
            show(img,show_size*2,"Texture Block")
    return img_arr

#--------------------------Feature  Extraction---------------------------------------

def get_pixel(img, center, x, y): 
      
    new_value = 0
      
    try: 
        # If local neighbourhood pixel  
        # value is greater than or equal 
        # to center pixel values then  
        # set it to 1 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
        # Exception is required when  
        # neighbourhood value of a center 
        # pixel value is null i.e. values 
        # present at boundaries. 
        pass
      
    return new_value 
   
# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y): 
   
    center = img[x][y] 
   
    val_ar = [] 
      
    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
      
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
      
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
      
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
      
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
      
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
      
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
      
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
       
    # Now, we need to convert binary 
    # values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val 
   

def get_LBP(sentence,show_steps=1,show_size=0.2):

    height, width = sentence.shape 


    # Create a numpy array as  
    # the same height and width  
    # of RGB image 
    img_lbp = np.zeros((height, width), np.uint8) 

    for i in range(0, height): 
        for j in range(0, width): 
            img_lbp[i, j] = lbp_calculated_pixel(sentence, i, j) 
    return img_lbp


class LocalBinaryPatterns:
    def __init__(self,nbins=256):
        self.nbins = nbins
    def describe(self, image):

        lbp = get_LBP(image)
        hestoImage,_ = histogram(lbp, nbins = self.nbins )
        return hestoImage


def read_directory(input_path):
    cases = []
    for file in os.listdir(input_path):
        cases.append(int(file))
    cases.sort()
    return cases

#-------------------------------Training and Classification-------------------------------------

def classify(clf,input_path,case,show_steps=1,show_size=0.2):
    desc = LocalBinaryPatterns(256)
    data = []
    gray_img = read_image(input_path+case+"/test.png")
    start = time.time()
    #print(input_path+case+"/test.png")
    binary_img = preprocess_img(gray_img,show_steps,show_size)
    binary_img,gray_img = remove_top(binary_img,gray_img,show_steps,show_size)
    large_texture_block = rearrange_image(binary_img,gray_img,show_steps,show_size)
    blocks = divide_image(large_texture_block,show_steps,show_size)
    for block in blocks:
        hist = desc.describe(block)
        data.append(hist)
    end = time.time()
    return np.bincount(np.around(clf.predict(data)).astype(int)).argmax(),end-start


def train(input_path,case,show_steps=1,show_size=0.2):
    desc = LocalBinaryPatterns(256)
    data = []
    labels = []
    total_time = 0
    for i in range(3):
        for  j in range(2):
            print(input_path+case+"/"+str(i+1)+"/"+str(j+1)+".png")
            gray_img = read_image(input_path+case+"/"+str(i+1)+"/"+str(j+1)+".png")
            start = time.time()
            binary_img = preprocess_img(gray_img,show_steps,show_size)
            binary_img,gray_img = remove_top(binary_img,gray_img,show_steps,show_size)
            large_texture_block = rearrange_image(binary_img,gray_img,show_steps,show_size)
            blocks = divide_image(large_texture_block,show_steps,show_size)
            for block in blocks:
                hist = desc.describe(block)
                labels.append(i+1)
                data.append(hist)
            end = time.time()
            total_time += end-start
    
    start = time.time()    
    model =  LinearSVC(C=300, random_state=42,max_iter=2000000000)
    model.fit(data, labels)
    end = time.time()
    total_time += end-start
    return model,total_time

#------------------TestCases generators and accuracy calculation---------------------------

def calculate_accuracy(classification, input_path, case, errors_log):
    f = open(input_path+ case + "/truth.txt", "r")
    
    result = int(f.read())
    f.close()
    if result == classification:
        return True
    else:
        errors_log.write("Wrong classification at case : " + case + "\n")
        return False


def read_forms_data():
    """
    this function read the forms.txt and extract the writer id and the form id that he wrote it.
    :return: dictionary of writers and the form that he has  written  key= form_id and value = writer_id
    """
    "change the path of the forms.txt here :)"
    path_forms_txt = 'F:/Tech/CUFE_CHS/Fall_2020/Pattern/Project/forms.txt'
    writers = {}
    reader = open(path_forms_txt)
    inputs = list(reader)
    for line in inputs:
        form_info = line.split(' ')
        if "#" not in form_info[0]:
            form_id = form_info[0]
            writer_id = form_info[1]
            writers[form_id] = writer_id

    return writers


def order_data_set():
    """
    this function call the read_forms_data() and collect all the forms written by the same author in one folder
    :return:
    """
    "change the direction of old data set forms here :)"
    old_path_data = 'F:/Tech/CUFE_CHS/Fall_2020/Pattern/Project/Data_Set/'
    writers_set = read_forms_data()
    for form in writers_set:
        new_path_data = 'ordered forms/'
        new_path_data += str(writers_set[form])
        form_name = str(form) + '.png'
        if not os.path.exists(new_path_data):
            os.makedirs(new_path_data)
            os.rename(old_path_data + form_name,
                      new_path_data + '/' + form_name)
        else:
            os.rename(old_path_data + form_name,
                      new_path_data + '/' + form_name)

    return

def filter():
    suitable = []
    path = "F:/Tech/CUFE_CHS/Fall_2020/Pattern/Project/Data_Set/ordered forms"
    for filename in os.listdir(path):
        i = len(os.listdir(path + "/" + filename))
        if (i >= 3):
            suitable.append(filename)
    suitable.sort()
    #print(suitable)
    return suitable,path


def generate_random_test_cases(num):
    new_path="F:/Tech/CUFE_CHS/Fall_2020/Pattern/Project/Data_Set/testcases/" + str(num) + "/"
    os.makedirs(new_path)
    suitable, dataset_path = filter()
    random.seed(datetime.now())
    random_choices = random.sample(suitable, 3)
    
    rand_index = random.randrange(0,3)
    test_images = []
    for i, choice in enumerate(random_choices):
        this_path = new_path + str(i+1)
        os.makedirs(this_path)
        images = os.listdir(dataset_path + "/" + choice)
        random_images = random.sample(images, 3)
        
        for i in range(2):
            shutil.copy2(dataset_path + "/" + choice + "/" +  random_images[i] , this_path)
            os.rename(this_path + "/" + random_images[i], this_path + "/" + str(i + 1) + ".png")
        test_images.append(dataset_path + "/" + choice + "/"+ random_images[2])
    shutil.copy2(test_images[rand_index] , new_path)
    os.chdir(new_path)
    for file in glob.glob("*.png"):
        os.rename(new_path + "/"+ file, "test.png")

    f = open(new_path + "truth.txt", "w")
    f.write(str(rand_index + 1))
    f.close()
    print("Test case " + str(num) + " generated!")
    print("Truth is " + str(rand_index + 1))




def main():
    #arg_path = sys.argv[]
    input_path = ""
    if len(sys.argv) != 2:
        input_path = sys.argv[1] + "/testcases/"
    else:
        input_path = "testcases/"
    #input_path = sys.argv[0]
    #'F:/Tech/CUFE_CHS/Fall_2020/Pattern/Project/Data_Set/testcases/'
    show_steps = 0
    show_size = 0.2
    random.seed(1)
    cases = read_directory(input_path)
    # print("Cases is ", cases)
    acc_list = []
    classifications = open("results.txt", "w")
    errors_log = open("error.log", "w")
    timing = open("time.txt", "w")
    for i, case in enumerate(cases):
        try :
            print("Running Test Case", i)
            total_time = 0
            case = str(case)
            clf,train_time = train(input_path,case,show_steps,show_size)
            total_time += train_time
            y,test_time = classify(clf,input_path,case,show_steps,show_size)
            total_time +=  test_time
            timing.write(str(round(total_time,2))+"\n")
            classifications.write(str(y)+"\n")
            acc_list.append(calculate_accuracy(y, input_path, case,errors_log))
        except Exception as e:
            print(e)
            y = int((int(random.random()*100) % 3) + 1)
            classifications.write(str(y)+"\n")
            errors_log.write("Exception at case : " + case + "\n")

        
    acc_list = np.array(acc_list)
    print("Accuracy: ", (len(acc_list[acc_list == True])/ len(acc_list)) * 100, "%")
    classifications.close()
    errors_log.close()
    timing.close()

if __name__ == "__main__":
    main()

