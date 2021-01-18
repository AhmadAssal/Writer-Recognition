from datetime import datetime
import random
import os
import shutil
import glob,fnmatch
"""
this script made by mad max aka "SamyBahaa"
"""


def read_forms_data():
    """
    this function read the forms.txt and extract the writer id and the form id that he wrote it.
    :return: dictionary of writers and the form that he has  written  key= form_id and value = writer_id
    """
    "change the path of the forms.txt here :)"
    path_forms_txt = '/home/ahmad/Desktop/AlexRun/Pattern Recognition/Dataset/ascii/forms.txt'
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
    old_path_data = '/home/ahmad/Desktop/AlexRun/Pattern Recognition/Dataset/forms/'
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
    path = "/home/ahmad/ordered forms"
    for filename in os.listdir(path):
        i = len(os.listdir(path + "/" + filename))
        if (i >= 3):
            suitable.append(filename)
    suitable.sort()
    #print(suitable)
    return suitable,path


def generate_random_test_cases(num):
    new_path="/home/ahmad/Desktop/AlexRun/Pattern Recognition/Dataset/testcases/testcase_" + str(num) + "/"
    os.makedirs(new_path)
    suitable, dataset_path = filter()
    random.seed(datetime.now())
    random_choices = random.sample(suitable, 3)
    
    rand_index = random.randrange(0,3)
    test_images = []
    for i, choice in enumerate(random_choices):
        # TODO 0: Create a new folder
        this_path = new_path + str(i+1)
        os.makedirs(this_path)
        # TODO 1: copy images from original folder to new folder
        # TODO 1.1: get old folder 
        images = os.listdir(dataset_path + "/" + choice)
        # TODO 1.2: get three random images
        random_images = random.sample(images, 3)
        
        # TODO:1.3: copy 2 of them into a folder
        for i in range(2):
            shutil.copy2(dataset_path + "/" + choice + "/" +  random_images[i] , this_path)
            os.rename(this_path + "/" + random_images[i], this_path + "/" + str(i + 1) + ".png")
        test_images.append(dataset_path + "/" + choice + "/"+ random_images[2])
    # TODO 2: copy a random image as a test outside folders
    shutil.copy2(test_images[rand_index] , new_path)
    os.chdir(new_path)
    for file in glob.glob("*.png"):
        os.rename(new_path + "/"+ file, "test.png")

    # TODO 3: write the random image original location in a text file for accuracy testing later 
    f = open(new_path + "truth.txt", "w")
    f.write(str(rand_index + 1))
    f.close()
    print("Test case " + str(num) + " generated!")
    print("Truth is " + str(rand_index + 1))
#order_data_set()
#filter()
for i in range(0,100):
    generate_random_test_cases(i)