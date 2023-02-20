
The inspiration for this solution came from here:\
<br>
https://medium.com/cars24-data-science-blog/blur-classifier-image-quality-detector-7c1de5ff8e59
<br>

As you see, they just highlight their solution, but it was so interesting, I decided to implement it.\
I used the same dataset yes open/closed and created two blured versions of each image.

# What's been done:
  * From each 8x8 patch, spit the DCT output into low, medium and high speed frequencies.
  * Calculate mean, std, kurthisis, skewness, entropy and energy for each range. 
  * Average for the whole image.
  * run XGBoost classifier
  * Optimize model parameter - cross-validation and hyperparameter tuning with optuna. Use ROC AUC metrics for training.
  * Analyze precision, recal, confusion matrix and accuracy.

# The Results
  Are nothing to write home about :). I reached ~ 73% ROC. But with better dataset I am convinced this approach could work.

<br>
<br>

**The blurs are done with library**:

https://github.com/NatLee/Blur-Generator

I randomly choose two out three blurs with replacement and also randomly choose thei parameters out of chosen ones.\
The parameters are chosen so that the blurs as realistic as possible to the real cases.

**tools\2_blur_types_selection.py** shows how I chose the blur parameters.

**tools\3_image_dataset_creation.py** creates the image dataset.

**tools\4_feature_vectors_dataset_creation.py** splits the images into 8x8 patches and saves the dat for DCT processing.\
I am really proud of this script because I moved to torch, which allows GPU and I created all the patches with a single oneliner :).

**tools\5_feature_extraction_per_patch.py** does the heavy lifting of feature extraction,\
I had to use multiprocessing, otherwise it would have taken about a day.\
There were just about 11M patches for ~ 2700 images.

**tools\6_final_features.py** creates the final features: 18 features per image. \
THe data is saved in the data/features.csv which is the classifier input.




# How would I solve it:
  I would read more. There are 5-6 recent promising nets, which beat all classical approaches. 
  But I would also try mine approach with a beter dataset.



