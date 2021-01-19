def classify(clf,input_path : str , case : str , show_steps = 1 : int , show_size = 0.2 : float) -> int:
    """Classify the given input test case
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    input_path : str
    show_steps : int , optional
    show_steps : float , optional
    """
    desc = LocalBinaryPatterns(254, 3)
    data = []
    gray_img = read_image(input_path+case+"/test.png")
    binary_img = preprocess_img(gray_img,show_steps,show_size)
    binary_img,gray_img = remove_top(binary_img,gray_img,show_steps,show_size)
    large_texture_block = rearrange_image(binary_img,gray_img,show_steps,show_size)
    blocks = divide_image(large_texture_block,show_steps,show_size)
    for block in blocks:
        hist = desc.describe(block)
        data.append(hist)

    return np.bincount(np.around(clf.predict(data)).astype(int)).argmax()


def train(input_path : str , case : str , C_test = 100 : int , show_steps = 1 : int , show_size = 0.2 : float):
    """train the input images
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    input_path : str
    case : str
    C_test : int
    show_steps : int , optional
    show_steps : float , optional
    """
    desc = LocalBinaryPatterns(254, 3)
    data = []
    labels = []
    for i in range(3):
        for  j in range(2):
            gray_img = read_image(input_path+case+"/"+str(i+1)+"/"+str(j+1)+".png")
            binary_img = preprocess_img(gray_img,show_steps,show_size)
            binary_img,gray_img = remove_top(binary_img,gray_img,show_steps,show_size)
            large_texture_block = rearrange_image(binary_img,gray_img,show_steps,show_size)
            blocks = divide_image(large_texture_block,show_steps,show_size)
            for block in blocks:
                hist = desc.describe(block)
                labels.append(i+1)
                data.append(hist)

    model =  LinearSVC(C=C_test, random_state=42,max_iter=200000)
    model.fit(data, labels)
    return model

def calculate_accuracy(classification : int , input_path , case , errors_log ):
    f = open(input_path+ case + "/truth.txt", "r")
    
    result = int(f.read())
    f.close()
    if result == classification:
        return True
    else:
        errors_log.write("Wrong classification at case : " + case + "\n")
        return False