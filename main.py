from code.classifications import *
from Writer-Recognition.code.constants import *
from Writer-Recognition.code.image_preprocessing import *
from Writer-Recognition.code.utils import *




accuracies = open(accuracies_output, "w")
cases = read_directory(input_path)
acc_list = []

classifications = open(classifications_output, "w")
errors_log = open(errors_output, "w")
for i, case in enumerate(cases):
    case = str(case)
    try:
        clf = train(input_path,case,C_test,show_steps,show_size)
        y = classify(clf,input_path,case,show_steps,show_size)
        classifications.write(str(y)+"\n")
        acc_list.append(calculate_accuracy(y, input_path, case,errors_log))
    except:
#         y = int((int(random.random()*100) % 3) + 1)
#         classifications.write(str(y)+"\n")
        errors_log.write("Exception at case : " + case + "\n")

acc_list = np.array(acc_list)
print("Accuracy: ", (len(acc_list[acc_list == True])/ len(acc_list)) * 100, "%")
accuracies.write("Accuracy For C =  "+ str(C_test) + " = " +str((len(acc_list[acc_list == True])/ len(acc_list)) * 100)+ "%")
classifications.close()
errors_log.close()
