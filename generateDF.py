import os
import pandas as pd
import glob
def getDirOnly(path):
    return [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]


def generateTrain(root): # for me C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data
    colname = ["root", "MR_number", "slice_number", "localImPath", "localMaskPath"]
    stackList = []
    typePath = os.path.join(root, "CHAOS_Train_Sets\Train_Sets\MR")
        
    for numberDir in getDirOnly(typePath):
        for file in list(glob.glob(os.path.join(typePath, numberDir,"T2SPIR", "DICOM_anon",'*'))): 
            
            slice_number =  os.path.splitext(os.path.basename(file))[0]
            
            localImPath = os.path.join("CHAOS_Train_Sets\Train_Sets\MR", numberDir,
                                    "T2SPIR", "DICOM_anon", os.path.basename(file))
            localMaskPath = os.path.join("CHAOS_Train_Sets\Train_Sets\MR", numberDir,
                                    "T2SPIR", "Ground", os.path.splitext(os.path.basename(file))[0]+".png")
            row = [root, numberDir, slice_number, localImPath, localMaskPath]
            stackList.append(row)
                
    trainDF = pd.DataFrame(stackList)
    trainDF.columns = colname
    return trainDF

def generateTest(root): # for me C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data
    colname = ["root", "MR_number", "slice_number", "localImPath"]
    stackList = []
    typePath = os.path.join(root, "CHAOS_Test_Sets\Test_Sets\MR")
    for numberDir in getDirOnly(typePath):
        for file in list(glob.glob(os.path.join(typePath, numberDir,"T2SPIR", "DICOM_anon",'*'))): 
           # numberPathT2 = os.path.join(typePath, numberDir,"T2SPIR") #T2 images only
            
            localImPath = os.path.join("CHAOS_Test_Sets\Test_Sets\MR", numberDir,
                                    "T2SPIR", "DICOM_anon", os.path.basename(file))
            slice_number =  os.path.splitext(os.path.basename(file))[0]
            row = [root, numberDir, slice_number, localImPath]
            stackList.append(row)
                
    testDF = pd.DataFrame(stackList)
    testDF.columns = colname
    return testDF
            
if __name__ == "__main__":
    root = r"D:\IMT Atlantique\TAF\ProCom\rendu\ProCom-lightweight-network\data"
    train = generateTrain(root)
    train.to_csv(os.path.join(root, "train2D.csv"), index=False)
    test = generateTest(root)
    test.to_csv(os.path.join(root, "test2D.csv"), index=False)