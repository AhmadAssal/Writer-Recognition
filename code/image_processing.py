def crop_img(img) -> numpy dtype object:
    """Crops the image so that no white spaces are left on left,right,top and bottom
    Parameters
    ----------
    img : numpy dtype object
    """
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

def preprocess_img(img : numpy dtype object, show_steps = 1 : int, show_size = 0.2 : float) -> numpy dtype object :
    """Preprocess an image an returns a binarized version of the image
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    img : numpy dtype object
    show_steps : int , optional
    show_steps : float , optional
    """
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

def remove_top(binary_img : numpy dtype object, gray_img : numpy dtype object, show_steps = 1 : int, show_size = 0.2 : float) -> numpy dtype object,numpy dtype object:
    """Removes the top and bottoma area to isolate the handwriting
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    binary_img : numpy dtype object
    gray_img : numpy dtype object
    show_steps : int , optional
    show_steps : float , optional
    """

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


def merge_boxes(boxes : list) -> list:
    """Merges the overlapping contour boxes
    ----------
    boxes : list
    """
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

def get_sentences(binary_img : numpy dtype object , gray_img : numpy dtype object , show_steps = 1 : int , show_size = 0.2 : float) -> list,list:
    """Extract sentences from a given image
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    binary_img : numpy dtype object
    gray_img : numpy dtype object
    show_steps : int , optional
    show_steps : float , optional
    """
    
    original_binary = binary_img.copy()
    original_gray = gray_img.copy()    
    canny = cv2.Canny(binary_img, 200, 400)
    kernel = np.ones((1,7),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=40)
    
    if show_steps == 1:        
        show(dilate,show_size,"Dialted for Sentence Extraction")
    
    cnts,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

            if h > binary_img.shape[0]/5:
                continue
            cv2.rectangle(binary_img, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI_binary = original_binary[y:y+h, x:x+w]
            ROI_gray = original_gray[y:y+h, x:x+w]
            hor_sum = np.sum(ROI_binary,axis=1)
            if h > binary_img.shape[0]/3:
                binary_images.append(ROI_binary[0:int(ROI_binary.shape[0]/2),:])
                gray_images.append(ROI_gray[0:int(ROI_gray.shape[0]/2),:])
                order_list.append(y)
                binary_images.append(ROI_binary[int(ROI_binary.shape[0]/2):,:])
                gray_images.append(ROI_gray[int(ROI_gray.shape[0]/2):,:])
                order_list.append(y+1)
            else:
                binary_images.append(ROI_binary)
                gray_images.append(ROI_gray)
                order_list.append(y)

    sentences_binary = sort_lists(order_list,binary_images)
    sentences_gray = sort_lists(order_list,gray_images)
    return sentences_binary,sentences_gray

def get_horizontal_merge(binary_sentence : numpy dtype object , gray_sentence : numpy dtype object , show_steps = 1 : int , show_size = 0.2 : float) -> numpy dtype object,float:
    """Get the horizontally merged sentence
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    binary_img : numpy dtype object
    gray_img : numpy dtype object
    show_steps : int , optional
    show_steps : float , optional
    """
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



def rearrange_image(binary_img : numpy dtype object , gray_img : numpy dtype object , show_steps = 1 : int , show_size = 0.2 : int) -> numpy dtype object :
    """Get the horizontally merged sentence
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    binary_img : numpy dtype object
    gray_img : numpy dtype object
    show_steps : int , optional
    show_steps : float , optional
    """
    copy = np.zeros((binary_img.shape[0], binary_img.shape[1]))
    binary_sentences,gray_sentences = get_sentences(binary_img,gray_img,show_steps,show_size)
    contours, hierarchy = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    currentY = 150
    for i in range(len(binary_sentences)):
        sentence,avg_height = get_horizontal_merge(binary_sentences[i],gray_sentences[i],show_steps,show_size)
        ROI = copy[currentY:currentY+sentence.shape[0],0:sentence.shape[1]]
        ROI += sentence
        copy[currentY:currentY+sentence.shape[0],0:sentence.shape[1]] = ROI
        currentY += int(avg_height/2)
    copy = crop_img(copy)
    copy = copy.astype(np.uint8)
    if show_steps == 1:        
        show(copy,show_size*2,"Vertical Merge")
    return copy

def divide_image(image : numpy dtype object , show_steps = 1 : int , show_size = 0.2 : float) -> numpy dtype object :
    """Divide the image into 9 texture blocks
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    image : numpy dtype object
    show_steps : int , optional
    show_steps : float , optional
    """
    factor = 3
    height, width = image.shape
    img_arr = []
    w_3 =int(width/3)
    h_3= int(height/3)
    for i in range(9):
        rand_row = int((random.random()*image.shape[0]) %(image.shape[0] - 128))
        ran_column = int((random.random()*image.shape[1]) %(image.shape[1] - 256))
        img_arr.append(image[rand_row:rand_row+128 , ran_column:ran_column+256])
    if show_steps == 1:
        for img in img_arr:
            show(img,show_size*2,"Texture Block")
    return img_arr


def describe(image, eps=1e-7) -> numpy dtype object:
    """Gets 
    The show_steps, and show_size are used for debuging purposes and are optional and if not provided 1 and 0.2 are used respectively
    Parameters
    ----------
    image : numpy dtype object
    eps : float
    """
    numPoints = 254
    radius = 3

    lbp = feature.local_binary_pattern(image,numPoints,radius, method="default")
    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, numPoints + 3),
        range=(0, numPoints + 2))

    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return hist