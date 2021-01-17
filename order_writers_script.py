import os
"""
this script made by mad max aka "SamyBahaa"
"""

def read_forms_data():
    """
    this function read the forms.txt and extract the writer id and the form id that he wrote it.
    :return: dictionary of writers and the form that he has  written  key= form_id and value = writer_id
    """
    "change the path of the forms.txt here :)"
    path_forms_txt = 'forms.txt'
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
    old_path_data = 'forms/'
    writers_set = read_forms_data()
    for form in writers_set:
        new_path_data = 'ordered forms/'
        new_path_data += str(writers_set[form])
        form_name = str(form) + '.png'
        if not os.path.exists(new_path_data):
            os.makedirs(new_path_data)
            os.rename(old_path_data + form_name, new_path_data + '/' + form_name)
        else:
            os.rename(old_path_data + form_name, new_path_data + '/' + form_name)

    return


order_data_set()
