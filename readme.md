# ProCom-lightweight-network

Image segmentation using a lightweight UNet that benefits from knowledge distillation from a cumbersome UNet to perform segmentation of medical images and get as accurate segmentation as cumbersome UNet's segmentation.

As the dataset train2D.csv and test2D.csv have already been built and saved in the folder data, you just have to run the Knowledge distillation notebook to perform knowledge distillation.

Also, if you want to skip training part and only see results, you can use all models saves that are in the data folder (and of course skip, the training cells in the notebook).

### Generate from scratch
If you want to generate everything from scratch please follow the instructions below :  

First, download the CHAOS challenge datasets and put in in "data" folder
download links(download the 2 datasets):  
https://zenodo.org/record/3431873/files/CHAOS_Test_Sets.zip?download=1  
https://zenodo.org/record/3431873/files/CHAOS_Train_Sets.zip?download=1


Generate Dataframe with :

```
python3 generateDF.py
```

then train by indicating absolute path of the folder where you saved the 2 datasets:

```
python3 train.py <absolute path of data folder>
```
