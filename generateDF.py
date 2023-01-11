import os
import pandas as pd

def getDirOnly(path):
    return [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]


def generateTrain(root): # for me C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data
    colname = ["root", "imgNumber", "number_slices", "localImPath", "localMaskPath"]
    stackList = []
    typePath = os.path.join(root, "CHAOS_Train_Sets\Train_Sets\MR")
        
    for numberDir in getDirOnly(typePath):
        numberPathT2 = os.path.join(typePath, numberDir,"T2SPIR") #T2 images only
        number_slices = len(os.listdir(os.path.join(numberPathT2, "DICOM_anon")))
        
        localImPath = os.path.join("CHAOS_Train_Sets\Train_Sets\MR", numberDir,
                                   "T2SPIR", "DICOM_anon")
        localMaskPath = os.path.join("CHAOS_Train_Sets\Train_Sets\MR", numberDir,
                                   "T2SPIR", "Ground")
        row = [root, numberDir, number_slices, localImPath, localMaskPath]
        stackList.append(row)
                
    trainDF = pd.DataFrame(stackList)
    trainDF.columns = colname
    return trainDF

def generateTest(root): # for me C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data
    colname = ["root", "imgNumber", "number_slices", "localImPath"]
    stackList = []
    typePath = os.path.join(root, "CHAOS_Test_Sets\Test_Sets\MR")
    for numberDir in getDirOnly(typePath):
        numberPathT2 = os.path.join(typePath, numberDir,"T2SPIR") #T2 images only
        localImPath = os.path.join("CHAOS_Test_Sets\Test_Sets\MR", numberDir,
                                   "T2SPIR", "DICOM_anon")
        number_slices = len(os.listdir(os.path.join(numberPathT2, "DICOM_anon")))
        row = [root, numberDir, number_slices, localImPath]
        stackList.append(row)
                
    testDF = pd.DataFrame(stackList)
    testDF.columns = colname
    return testDF
            
if __name__ == "__main__":
    root = r"C:\Users\nampo\Downloads\Data"
    train = generateTrain(root)
    train.to_csv(os.path.join(root, "train.csv"), index=False)
    test = generateTest(root)
    test.to_csv(os.path.join(root, "test.csv"), index=False)