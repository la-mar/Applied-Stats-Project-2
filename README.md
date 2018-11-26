# PROJECT 2 SPRING 2018:

KOBE BRYANT SHOT SELECTION !!!



## OVERVIEW:
Kobe Bryant marked his retirement from basketball by scoring 60 points in his final game as a member of the Los Angeles Laker team on Wednesday, April 12, 2016. Starting to play professional basketball at the age of 17, Kobe earned the sport’s highest accolades throughout his long career.  Using 20 years of data on Kobe's shots made and shots missed, can you predict which shots will be successful?

## DATA:
The original data set contains the location and circumstances of every shot attempted by Bryant during his 20-year career.

__Your task is to predict whether the basket went in (shot_made_flag = 1) or missed (shot_made_flag = 0).  The data for estimation is in project2Data.xlsx.__

For this exercise, 5000 of the shot_made_flags have been removed from the original data set and are shown as missing values in the project2Pred.xlsx file.  These are the test set shots for which you must submit a classification. You are provided a sample classification file, project2Pred.xlsx with the shot_ids needed for your predicted classification.  Provide you predicted classifications in this file and submit both your paper and the prediction file. I have the actual values of the shot_made_flag for these missing shot_ids and will evaluate the classifications.  Your goal is to provide the best predictions possible.

__Turn in paper and predictions__

Each group is on the honor system to not use any information outside of the dataset to predict each of the missing shot flags.
 
## DATA CONTINUED

The field names are given below (Data descriptions are available in Kaggle):

action_type
combined_shot_type
game_event_id
game_id
lat – court location identifier (latitude)
loc_x - court location identifier (x/y axis)
loc_y- court location identifier (x / y axis)
lon - court location identifier (longitude)
minutes_remaining – (in period)
period
playoffs
season
seconds_remaining
attendance
avgnoisedb – avg noise in arena (decibels)	shot_distance
shot_made_flag (this is what you are predicting)
shot_type
shot_zone_area
shot_zone_basic
shot_zone_range
team_id
team_name
game_date
matchup
opponent
shot_id
arena_temp (oF)

## DELIVERABLE:
Students will submit a paper with an 8 page limit with a separate Appendix up to 5 pages.  Code should be in a second appendix and can be as long as necessary.   A separate file with predicted classifications also should be submitted.

## PAPER REQUIREMENTS

# Introduction

## Data Description

### Exploratory Data Analysis

•	__potential transformations__
•   __Normaility__
•	__outliers__
•	__multicollinearity__

### Build Models

Build models to provide arguments and evidence for or against the propositions below:

#### Answer these questions with our results:

•	The __odds of Kobe making a shot decrease with respect to the distance he is from the hoop__.  If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

•	The __probability of Kobe making a shot decreases linearly with respect to the distance he is from the hoop__.    If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

•	The relationship between the __distance Kobe is from the basket and the odds of him making the shot is different if they are in the playoffs__.  Quantify your findings with statistical evidence one way or the other. (Tests, CIs, plots, etc.) 

##### Build a predictive model to classify shots as missed or made.  You should produce at least 1 of each type of model:
###### A logistic regression model.
###### A Linear Discriminant Analysis (LDA) model.

## Evaluation:
Compare each competing models using:
    - AUC
    - Mis-Classification Rate
    - Sensitivity
    - Specificity and objective / loss function

The __log loss function__ of the model should be used to assess the model fit:
    - Where N is the total number classifications, yi is the shot_made_flag and pi is the probability from the model of each outcome (shot made or shot missed.)

## ASSESSMENT / EVALUATION:

Good papers traditionally have the following characteristics:
1.	They are presented in an organized, neat and consistent fashion. (Labeled plots, figures and tables, consistently formatted, indented and labeled headers and sub headers, etc.)  Given that each group has 3 members, the paper should only have one look and feel.  Titles, headers, sub headers, figures, tables, etc. should all look the same and have numbering that is consistent.
2.	There are no typos, misspelled words, grammatical mistakes, etc.
3.	They use a variety of methods.
4.	Creative methods are used.
5.	They have input from all group members and are developed iteratively over time as opposed to all at once such as the night before.

The group with the lowest log loss score will be awarded an additional 3 points for the project.

## SOFTWARE AND METHODS:
You may use any software and must use only the methods we have studied thus far in the course.  That being said, you can use innovative techniques inside of those methods like model averaging, cross validation or creating new variables from the ones in the data set.  If you have any questions about this please let me know and we can discuss your ideas.

