> The material in Dr. Mystery's Lab Guide is partially derived from "[Whitaker Lab Project Management](https://github.com/WhitakerLab/WhitakerLabProjectManagement)" by Dr. Kirstie Whitaker and the Whitaker Lab team, used under CC BY 4.0. Dr. Mystery's Lab Guide is licensed under CC BY 4.0 by [Lars Andreas Metsälä Wulff, Regitze Julie Sydendal and Nikolette Zoe Pedersen].


# Andreas, Regitze and Nikolette's Weekly Meeting Notes

* [Tuesday, 6th February 2024](#date-6th-february-2024)
* [Tuesday, 13th February 2024](#date-13th-february-2024)
* [Tuesday, 20th February 2024](#date-20th-february-2024)
* [Tuesday, 27th February 2024](#date-27th-february-2024)
* [Tuesday, 5th March 2024](#date-5th-march-2024)
* [Tuesday, 12th March 2024](#date-12th-march-2024)
* [Tuesday, 19th March 2024](#date-19th-march-2024)
* [Tuesday, 2nd April 2024](#date-2nd-april-2024)
* [Tuesday, 9th April 2024](#date-9th-april-2024)
* [Tuesday, 15th April 2024](#date-15th-april-2024)
* [Tuesday, 30th April 2024](#date-30-April-2024)
* [Tuesday, 7th May 2024](#date-7-May-2024)


### Date: 6th February 2024


#### Who did you help this week?

* N/A 
  
#### What helped you this week?

* The Purrlab Notion
  
#### What did you achieve?

* We have set up the GitHub. We have talked about expectations during the process in the group.
  * We have got our working technologies up and running (Kanban)
  * Introduced first year project to Andreas 
  
#### What did you struggle with?

* N/A
  
#### What would you like to work on next week?

* We would like to look at the submissions and filter through them
  * Present the ones we found to Veronika

#### Where do you need help from Veronika?

* It would be nice with deadline from Veronika, so we can get a better overview
* Find out which features are interesting to look at (for the submissions) and what requirements/guidelines we should look for
* How should we share this with you? Should you be a collaborator?
* Is it okay we use ACL overleaf template
* When is it recommended we start writing the project?
  
#### What are the agreements after this meeting? (to fill in after the meeting)

* Summary on what we want to use
  * Conceptualise the experiment (what data, how? Like in the DTU-paper they add people in intervals, maybe do the same for the Fitzpatrick scale) 
* Next thing would be to think about the experimental format. What would we compare the model on, test on, evaluate etc.
* Upload submission evaluation to GitHub
* Write what we can write in Overleaf



### Date: 13th February 2024


#### Who did you help this week?

* Andreas, helped with technologies in relation to machine learning

#### What helped you this week?

* DTU research paper
* Pad-Ufes-20 reseach paper
  
#### What did you achieve?

* Graded submissions, and chose best candidates
* Explored our dataset
* Started experimental setup
* Began our bachelor-paper
* Gained a common understanding of research papers
  
#### What did you struggle with?

* How to split our data
  
#### What would you like to work on next week?

* Clean data
* Get code from selected submissions working
* Continue writing the bachelor-paper
* Settle on a state of the art CNN to continue working with

#### Where do you need help from Veronika?

* Are we interested in looking on only sex, 
colour, or both?
  * Pad-Ufes-20 are mostly fair skinned, would it even make sense to look at skin colour? There are still variations in the Fitzpatrick scale, although minor.
  * If we do not work on skin colour, should we consider Isic instead?
* Do we agree that this project is disease/cancer moreso than healthy/cancer?
* If we split our test data like they did in the DTU study (5 datasets consisting of $4\times25$ (f/a, f/h, m/a, m/h)), we end up with almost no datapoints without cancer in our training data. Should we do smaller datasets or is it okay to have almost no healthy datapoints in our training data. Is there another option?
* 7-point checklist as well as ABCD(E)?
  
#### What are the agreements after this meeting? (to fill in after the meeting)

* Look at a CNN (papers: pad-ufes (dtu))
* use multiple feature extractions methods, compare if they get the same asymmetry ect - use both ABCDE and 7-point checklist 
* Look at feature dist for the extracted features
* We will start with looking at sex - later skin color
* Remember to seperate the at patient-id, not lesions ID



### Date: 20th February 2024


#### Who did you help this week?

* N/A
  
#### What helped you this week?

* A classmate (Erling)
  
#### What did you achieve?

* Split the data into test and train, done like in the DTU-paper
* Found a CNN (ResNet-50)
* Made masks for all the images (tweaked group 9's code)
* Got group 9s code to run, we now have a csv with the extracted features for one group (color, asymmetry, compactness, convexity)
  
#### What did you struggle with?

* the hand crafted methods
* the masking algorithms are not great
  
#### What would you like to work on next week?

* we are meeting in the weekend:
  * get the hand crafted methods to work, so we can compare the feature extraction - look at submissions from 2022 if the problems continue
  * split the images, like we have split the metadata
  * would be nice to have set up the CNN

#### Where do you need help from Veronika?

* Better algorithms to mask the data? we use gaussion mask, it doesnt seem like any of the students have a good automated process for it
* We have chosen ResNet. We just looked at the best performing CNNs in the pad-ufes paper, is that ok?
* It seems like we can import ResNet50 from keras, does that seem right?
* My (Nikolette) sister is defending her Master's degree between the 14th-30th june and Andreas is doing a triathlon the 1st june. could it be possible to already wish for days inbetween that period to defend our bachelor?
  
#### What are the agreements after this meeting? (to fill in after the meeting)

* Make feature distribution
* Research and implement ResNet-50



### Date: 27th February 2024


#### Who did you help this week?

* N/A
  
#### What helped you this week?

* Dataset with masks we got from Veronika
  
#### What did you achieve?

* Feature distribution for two student submissions (ABCD(E)) (with the new data)
  * ![group4_feature](https://github.itu.dk/storage/user/4277/files/6fe663f7-3f74-4fbd-a39f-ec7bd3fdcc33)
  * ![group9_features](https://github.itu.dk/storage/user/4277/files/97d82764-a649-485d-9e88-261f446dbc08)
* Ran just a simple lr for the two student codes

* Split Pad-Ufes data up into training and test sets, with the correct distribution
* Got a working version of ResNet-50
  
#### What did you struggle with?

* Adjusting to the new data
* Working with new technology
  
#### What would you like to work on next week?

* Feature distribution from two additional student submissions
  * 7 point scale
  * Another ABCD(E)
* Get ResNet training on our dataset, instead of random images
* Finish training and test splits
* If we get student masks, get feature distribution of Pad-Ufes

#### Where do you need help from Veronika?

* Questions to Pad-Ufes Dataset
  * Mulitple patients have same lesion ID in Pad-Ufes?
    * <img width="1393" alt="data" src="https://github.itu.dk/storage/user/4281/files/5170563d-1b36-4a22-9e61-2eb40357d5fe">
  * Different lesion ID but same image?
    * <img width="532" alt="lesion" src="https://github.itu.dk/storage/user/4281/files/3e1e40ac-0891-4ea6-8dc3-4faf8c8af7a1">)
  * We have 5 different test set. Do we need to split the same patient into different sets?
    * <img width="93" alt="same" src="https://github.itu.dk/storage/user/4281/files/a9ddd934-d09a-40a7-9db3-c013c2859fde">
* Code is not optimized, but works  perfectly well for our purpose. Is that a problem?
  
#### What are the agreements after this meeting? (to fill in after the meeting)

* Regarding the feature distribution: dont use avg red, green ect
  * Make scatter plot with features, and color the classes a color so they are easy to group
* Make pdf regarding curious things in the dataset
* Change the lesion ID
* Make another feature dist, for 7-point-scale and another ABCDE


### Date: 5th March 2024


#### Who did you help this week?

* N/A


#### What helped you this week?

* Response from PAD-UFES maker
  
  
#### What did you achieve?

* Make feature dist. for two student codes:

  * Scatter plot of group 9 features, different color values
  * ![feature9_all_color](https://github.itu.dk/storage/user/4277/files/be508fa2-0c37-4bc0-81c5-e90af444c51d)

  Scatter plot of group 9 different color values
  * ![feature9_allpng](https://github.itu.dk/storage/user/4277/files/1f62f94f-58c8-464e-8e07-989f1310849b)

  Scatter plot of group 4 features, compactness, asymmetry, convexity
  * ![feature4_all_scaled](https://github.itu.dk/storage/user/4277/files/67f54986-8cc2-4289-8286-41360c54ece5)

* CNN runs with our data
  

#### What did you struggle with?

* Figure out how to handle errors in our data set/fixing the mistakes
  

#### What would you like to work on next week?

* Get another student code ABCDE and a feature distribution for 7-point scale
* Set up experiements for new real data
  

#### Where do you need help from Veronika?

* Talk through how we will actually do the the increment of number of women in the data set when training, would be nice.
* We've fixed all 45 instances of shared lesion_ids in our dataset. Is that enough? Should we do more to the dataset, or should we just keep these limitations in mind, when writing the report?
  

#### What are the agreements after this meeting? (to fill in after the meeting)

* for the feature distribution, we should just take the biggest diseases
* make folders in latex (we already did)
* with a separation, we can measure how well the classes are separated (don't do that on all the data we have, as we’ll be separating on things we have seen
* try two models, one with metadata (like itching) and one without


### Date: 12th March 2024


#### Who did you help this week?

* N/A
 
  
#### What helped you this week?

* First year students who made the masks

 
#### What did you achieve?

* More feature distribution
  * group 9:
  * ![raindrop_group9](https://github.itu.dk/storage/user/4277/files/517897ba-ea39-479a-9f25-e4745491bc2a)
  * group 4:
  * ![raindrop_group4](https://github.itu.dk/storage/user/4277/files/de833eca-2d7a-4ec7-9aac-97947aaf68ee)
  
* Got the CNN to work with our data (again)
* Started on getting the 7-point-scale-code to run
* Gathered the masks from github


#### What did you struggle with?

* Took a long time gather the masks from the githubs
* Getting ptitprince to work (rain-cloud)

  
#### What would you like to work on next week?

* Refine CNN
* 7-point-scale distribution
* Would be really nice to run the first experiment on our real data, now that we have the masks


#### Where do you need help from Veronika?

* Brainstorm on fitzpatrick scale, the lesions are distributed like this: 1: 111, 2: 616, 3: 265, 4: 35, 5: 9, 6: 1


#### What are the agreements after this meeting? (to fill in after the meeting)

* Look at more related literature that looks at skin color, e.g. this website has some data (don’t necessarily use this data, but get some inspo from the website https://ddi-dataset.github.io/index.html)
* Make marginal plots: Marginal plots (related to raincloud) → seaborn
* Send the gathered mask data to Veronika once we’re done
* Maybe look here for metrics: https://metrics-reloaded.dkfz.de/


### Date: 19th March 2024


#### Who did you help this week?

* N/A
  
  
#### What helped you this week?

* First year students who made the masks
* Diverse Dermatology papers


#### What did you achieve?

* New feature distrubution with our real data 
  * group 4:
  * ![image](https://github.itu.dk/storage/user/4277/files/1b5ac8c3-8230-4fed-aba5-d8ccb33f9184)
  * ![image](https://github.itu.dk/storage/user/4277/files/e9f15235-5c97-4703-b817-b2a4e1ca3dda)



  * group 9:
  * ![image](https://github.itu.dk/storage/user/4277/files/3430a8f8-2005-450f-a887-c52740c4d8bc)
  * ![image](https://github.itu.dk/storage/user/4277/files/1328ff72-aa0d-4680-a210-7ae1037560a8)

 
* 7-point-check can run 
* Completed gathering the masks
* Confusion matrix from CNN


#### What did you struggle with?

* lack of masks from one group, and csv-json-masks from another group

  
#### What would you like to work on next week?

* Next week is easter and we will not be able to do too much work. But the week after easter we need to get results from our first experiments...


#### Where do you need help from Veronika?

* We are missing 300 masks from the students, out of 2300, should we wait for those or just accept that we cannot use those?
* Does it make any difference whether we use the classifiers from the student submission or classifiers we make ourselves?
* DDI dataset is a diverse dataset in regards to the Fitzpatrick scale consisting of 656 images. Would it make sense to use this dataset as test data for the models trained on Pad-Ufes-20? The diseases are not the same but perhaps something interesting could still be gathered by training models only on light skinned vs diverse and testing on this diverse dataset?
* Another dataset we found is the Fitzpatrick 17k. This contains 16,577 images, all anotated with Fitzpatrick values. However, the anotation is crowd-sourced instead of made by certified dermatologists. A dermatologist reviewed 3% of the dataset and confirmed that 69% are annotated correctly, while only 3.4% is distinctly wrong. Is this dataset useable for project in any capacity?
* In regards to Pad-Ufes-20, would it be possible to perform image processing to increase the amount of training data? For instance, in the Fitzpatrick 17k paper, they mention:
  * randomly resizing images to 256 pixels by 256 pixels 
  * randomly rotating images 0 to 15 degrees 
  * randomly altering the brightness, contrast, saturation, and hue of each image 
  * randomly flipping the image horizontally or not
  * center cropping the image to be 224 pixels by 224 pixels
  * normalizing the image arrays by the ImageNet means and standard deviations.


#### What are the agreements after this meeting? (to fill in after the meeting)

* We’ll receive masks this friday, and if they’re not sufficient, continue without them
* Make the non-deep learning classifier ourselves (knn, logistic regression, decision tree are viable options)
* Use the Fitzpatrick dataset as an extra test set for when we want to do experiments on skin colour
* Do use augmentations, but use a set of augmentations (inspo from Fitzpatrick 17k paper)


### Date: 2nd April 2024

#### Who did you help this week?

* N/A
 

#### What helped you this week?

* LabelStudio
  
  
#### What did you achieve?

* We have been on vacation, but we managed to make the remaning masks that we need.

  
#### What did you struggle with?

* Time used on masks

  
#### What would you like to work on next week?

* Run first experiments, now that we have our data
* Make a flow-chart of the data sets 
  

#### Where do you need help from Veronika?

* N/A
  
#### What are the agreements after this meeting? (to fill in after the meeting)

* Mention in the bachelor thesis that the masks created for the dataset are imperfect, as they are created by us and first-year students without much experience in neither dermatology nor label-studio.


### Date: 9 April 2024

#### Who did you help this week?

* N/A
  
  
#### What helped you this week?

* ADNI github
  
  
#### What did you achieve?

* Understand the way the splits actually are done
* Splitted our data with the same way they have, by using snippets from their code. This is now reproducable
* Flowchart
  * ![430091853_267413116442438_6154154770756102163_n](https://github.itu.dk/storage/user/4277/files/087075fd-55ba-456b-ad24-cb253a38045c)

  
#### What did you struggle with?

* We spent a lot of time reading and understanding their script

  
#### What would you like to work on next week?

* EXPERIMENTS 
 

#### Where do you need help from Veronika?

* In the research paper and code, the number 379 is mentioned as the data not used in the test sets. We cannot figure out how they got that number though, when we look at the numbers we know about their data. Should we write an email to them? 
* is sklearn logistic regression fine? or should we use pytorch (and other methods from their)
* How many times should we test?
  
#### What are the agreements after this meeting? (to fill in after the meeting)

* Write an email to the authors of the paper
* Use SKlearn for logistic regression

  
 ### Date: 15 April 2024

#### Who did you help this week?

* N/A

  
#### What helped you this week?

* N/A

  
#### What did you achieve?

* Features are split with the data now
* LR is ready to go for the experiments, mlflow is set up
* Written an email to the authors of the paper
  
  
#### What did you struggle with?

* Studentcode + splitting the data/re-writing a bit
* Imbalanced classes, we need to adjust for
 
 
#### What would you like to work on next week?

* Next week we would like to submit the draft, so work on everything


#### Where do you need help from Veronika?

* Good ways to handle imbalanced classes? Because test will probably be effected by this, so we need to counteract a bad test accuracy
* Should we combine ABCDE and 7-point check list? So we will have 125 LR models with ABCD features and 125 LR models with 7-point check list features
* They explain that they have made test-set that are equally balanced, but in their figures they have split up the test sets into female/male?
  * <img width="436" alt="image" src="https://github.itu.dk/storage/user/4277/files/c54c45a4-8a94-469a-83bc-fb8834085ee8">

  
#### What are the agreements after this meeting? (to fill in after the meeting)

*
  *
  
  ### Date: 30 April 2024

* [Tuesday, April 2024](#date-30-April-2024)

#### Who did you help this week?

* N/A
  * 
  
#### What helped you this week?

* Veronikas feedback

  
#### What did you achieve?

* Last last week we wrote the draft! The latest week we prioritesed our "negleted courses" (we also anticipated we would wait some days before getting feedback), which suffered under the week we wrote the draft:)
  
#### What did you struggle with?

* We are trying to make sure we understand everything in the Petersen et al correctly
  *
  
#### What would you like to work on next week?

* Make sure we understand the Petersen et al different significant testing fully, so we know we are doing it the correct way/re-do them the correct way
* Get balanced classes for the CN, so it hopefully perform better

#### Where do you need help from Veronika?

* We need to make sure we understand the Petersen et al different significance tests correctly, we are confused about:
  * They make both a t-test (which we think they use for determining if the slope is significant) and a Wilcoxon-tests (for seeing if the performance of the models, is LR better than CNN, are significant). Is this the correct understanding? picture:
    * <img width="645" alt="image" src="https://github.itu.dk/storage/user/4277/files/6dc7d99c-771b-47dd-895b-12b6cd932731">
    * <img width="628" alt="image" src="https://github.itu.dk/storage/user/4277/files/1f8d4b42-76a3-4e4c-9540-b99d11308ccb">
    * But, we cannot make sense of the fact that in their code, when they say they do the t-test they use a Wilcoxon-test? Their code:
    * <img width="769" alt="image" src="https://github.itu.dk/storage/user/4277/files/9e0f2d1c-9b23-4b2a-bef3-b5da65e54bbe">

    * When they make the plots and fit the regression-lines and get the p-value, we assume this is by doing a t-test. This is also what we have tried to do, their code:
    * <img width="780" alt="image" src="https://github.itu.dk/storage/user/4277/files/b86e650d-61a2-41b9-91bd-9fb82a3486b1">
    * Sorry about the random screenshots, maybe it will make more sense when we talk about in the meeting:)



  * They correct both of these tests with Bonferroni. They use a factor of 8 for the t-test, and they use a factor of 2 for the Wilcoxon
  * When they correct with bonferroni it seems like they do not correct the alpha, but they corrected the p-value with multiplying with the number of tests - atleast we think that is what they do
    * <img width="626" alt="image" src="https://github.itu.dk/storage/user/4277/files/081fb90a-9b82-4d46-ba5d-21046a88ec43">

* How should we balance the classes for the CNN?
  * We have a couple of different options for how to balance the data for the CNN as we did with SMOTE for LR. Basically:
    * Remove cancerous data. Since the data sets are bigger than LR, it would perhaps be feasible to simply remove data in order to balance. This might not be enough on its own though, and would we just pick data at random to remove?
    * Transforming and augmenting the non-cancerous data to "create more data". This is possible with PyTorch.
    * Using DCGAN to generate syntetic images for the CNN, similar to SMOTE for LR. We're not sure yet how effective this will be since our dataset is rather small, and it seems a big investment if it ends up being unusable.

* Should we do skin-color? We think have time to do it, but it might not make sense?
  * We would probably use the same methods, as the method we choose for balancing the classes for the CNN, to more get data of the darker skintones.
  * <img width="776" alt="image" src="https://github.itu.dk/storage/user/4277/files/d405d069-215d-4ee6-8949-bffe67afc79e">
  * We imaging that we could split the skin-tones in two, 1-3 and 4-6, then keep adding the skin tones 4-6 in ratios of 25%. Its a bit simplifiyed, but it would allow us to keep it binary with just like sex. 
  


#### Other
* We are going to Fejø on the 10-15th May to finish the bachelor thesis, so sadly we cannot go to hear about Víctor M. Campellos project the 13th of May:)
  
#### What are the agreements after this meeting? (to fill in after the meeting)
* Continue with the t-test and Wilcoxon test as we already did: The t-test has to do with the regression line and Wilcoxon has to do with the classifiers and whether they were different. For regression this is a standard way, and Wilcoxon is also standard because it’s non parametric
* Change up the way we use the Bonferroni correction, so divide the alpha with the number of tests instead of multiplying the p-values: Bonferroni, its the same whether we multiply. Don't multiply the p-values, just use the threshold instead, let the p-values stay the same. if you divide you can get into underflow problems
* Balance the classes by resampling: Balancing the classes, you should upsample, by resampling, but augmenting is fine with a little bit. maybe augment the duplicates. DON'T GO for the completely synthetic data
* Do a skin colour experiment: Skin colour could be fine with the way we proposed it. seems like a nice experiment. it’s kind of cheap, by keeping it binary

  *

### Date: 7 May 2024

* [Tuesday, May 2024](#date-7-May-2024)

#### Who did you help this week?

* N/A
  
  
#### What helped you this week?

* N/A
  
  
#### What did you achieve?

* Augmented data for CNN
* Added so that our splitting method works for augmented data
* We found a lot of mistakes and bugs, which we have corrected
  * We no longer have augmented data in our test classes
  * Train sets now contain augmented data, which allows for longer sets, but does not contain the augmented 'version' of the image used in corresponding test
  * We fixed an edge case where a data leak could occur, because patients with both cancerous and non cancerous lesions could end up having data in both test and corresponding train/tets val because of the way we previously split the data
* We can now test for female and male separately
* Our CNN is much better
* Feature extraction on images and augmented images is currently running and we are awaiting results
  
#### What did you struggle with?

* SMOTE for LR might be causing problems, we will do feature extraction on augmented images to avoid using it (which is a bit sad, we spent a lot of time on it, but oh well)
* Alot of new bugs kept showing up, but we are hopeful that we are past the worst of it
  
#### What would you like to work on next week?

* Submit the bachelors
* We hope to be able to do fitzpatrick, but it seems a bit more complicated than we thought. We will spend some hours and reasses whether the progress we are able to make in that time justifies continuing the work

#### Where do you need help from Veronika?

* Should we refere to the students papers?
* How much should we write about the students handcrafted features? Like explain how each of them works?
* Are we supposed to be able to "recite" the reserach papers we have refered to?
* Now that we both have smote and feature extraction from augmented images as a way to oversample data for the LR model, does it makes sense to include the comparisons that we are going to make between the two in the final paper? Or should we just include the optimal one of the two and showcase the results of that one?
* We think Smote might not be working, because we are not just oversampling a class but we are doing with a specific feature in mind (sex), could that be why? 
* How should justify choosing the final features for the LR? A combination of feature imporatance and the distrubtion plots we have made?
  * ![image](https://github.itu.dk/storage/user/4277/files/290e6576-55d5-4b92-82d8-ea4065bb337c)
* We want to write about all the considerations we have taken when splitting the data, considerations Petersen did not have to take, since we have a patient with mulitple lesions and such, because that has become a thing we have used a lot of time on, but is it releveant?
   
