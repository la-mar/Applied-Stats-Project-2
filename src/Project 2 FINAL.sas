/* MSDS 6371 - Applied Statistics  */
/* Allen Crane and Brock Friedrich */
/* Kobe Bryant Shot Selection      */
/* November 2018                   */

/* import data */
proc import datafile="c:\users\allen\documents\smu data science\MSDS 6372 - Applied Statistics\project 2\project2Data.csv"  
          dbms=dlm out=train replace;
     delimiter=',';
     getnames=yes;
run;

/* print data */
proc print data=train (obs=10);
run;

/* investigate data for variable types*/
proc contents data=train;
run;

/* look for missing numeric data - may need to impute */
proc means data = train n nmiss;
  var _numeric_;
run;

/* investigate means of training data and any missing values */
proc means data = train n nmiss;
run;

/* univariate data analysis */
proc univariate data = train;
var season;
run;

/* note that certain "season" fields are missing */
Data _season;
    set train;
    where missing (season);
run;

/* print "season" data, where "season" is missing */
proc print data=_season (obs=200);
run;


/* more univariate data analysis */
ods graphics on;
proc univariate data = train plot;
var recId 
game_event_id 
game_id 
lat 
loc_x 
loc_y 
lon 
minutes_remaining 
period 
playoffs 
season 
seconds_remaining 
shot_distance 
shot_made_flag 
team_id 
game_date 
shot_id 
attendance 
arena_temp 
avgnoisedb 
;
run;
ods graphics off;

/* create a time-based variable, concatenating Period, minutes remaining, and seconds remaining, in descending order. This one is for Periods remaining... */
data train2;
  set train;
      if period = 1 then periods_remaining2 = "14";
      else if period = 2 then periods_remaining2 = "28";
	  else if period = 3 then periods_remaining2 = "42";
	  else if period = 4 then periods_remaining2 = "57";
	  else if period = 5 then periods_remaining2 = "71";
	  else if period = 6 then periods_remaining2 = "85";
	  else if period = 7 then periods_remaining2 = "99";
	  else periods_remaining2 = period;
run;

/* print data */
proc print data=train2 (obs=10);
run;

/* Minutes remaining... */
data train2;
  set train2;
      if minutes_remaining = 11 then minutes_remaining2 = "99";
      else if minutes_remaining = 10 then minutes_remaining2 = "90";
	  else if minutes_remaining = 9 then minutes_remaining2 = "81";
	  else if minutes_remaining = 8 then minutes_remaining2 = "72";
	  else if minutes_remaining = 7 then minutes_remaining2 = "63";
	  else if minutes_remaining = 6 then minutes_remaining2 = "54";
	  else if minutes_remaining = 5 then minutes_remaining2 = "45";
	  else if minutes_remaining = 4 then minutes_remaining2 = "36";
	  else if minutes_remaining = 3 then minutes_remaining2 = "27";
	  else if minutes_remaining = 2 then minutes_remaining2 = "18";
	  else if minutes_remaining = 1 then minutes_remaining2 = "09";
	  else if minutes_remaining = 0 then minutes_remaining2 = "00";
	  else minutes_remaining2 = minutes_remaining;
run;

/* print data */
proc print data=train2 (obs=10);
run;

/* Seconds remaining... */
data train2;
  set train2;
      if seconds_remaining = 59 then seconds_remaining2 = "99";
      else if seconds_remaining = 58 then seconds_remaining2 = "97.3";
	  else if seconds_remaining = 57 then seconds_remaining2 = "95.6";
	  else if seconds_remaining = 56 then seconds_remaining2 = "94";
	  else if seconds_remaining = 55 then seconds_remaining2 = "92.3";
	  else if seconds_remaining = 54 then seconds_remaining2 = "90.6";
	  else if seconds_remaining = 53 then seconds_remaining2 = "88.9";
	  else if seconds_remaining = 52 then seconds_remaining2 = "87.3";
	  else if seconds_remaining = 51 then seconds_remaining2 = "85.6";
	  else if seconds_remaining = 50 then seconds_remaining2 = "83.9";
	  else if seconds_remaining = 49 then seconds_remaining2 = "82.2";
	  else if seconds_remaining = 48 then seconds_remaining2 = "80.5";
      else if seconds_remaining = 47 then seconds_remaining2 = "78.9";
	  else if seconds_remaining = 46 then seconds_remaining2 = "77.2";
	  else if seconds_remaining = 45 then seconds_remaining2 = "75.5";
	  else if seconds_remaining = 44 then seconds_remaining2 = "73.8";
	  else if seconds_remaining = 43 then seconds_remaining2 = "72.2";
	  else if seconds_remaining = 42 then seconds_remaining2 = "70.5";
	  else if seconds_remaining = 41 then seconds_remaining2 = "68.8";
	  else if seconds_remaining = 40 then seconds_remaining2 = "67.1";
	  else if seconds_remaining = 39 then seconds_remaining2 = "65.4";
	  else if seconds_remaining = 38 then seconds_remaining2 = "63.8";
	  else if seconds_remaining = 37 then seconds_remaining2 = "62.1";
	  else if seconds_remaining = 36 then seconds_remaining2 = "60.4";
	  else if seconds_remaining = 35 then seconds_remaining2 = "58.7";
	  else if seconds_remaining = 34 then seconds_remaining2 = "57.1";
	  else if seconds_remaining = 33 then seconds_remaining2 = "55.4";
	  else if seconds_remaining = 32 then seconds_remaining2 = "53.7";
	  else if seconds_remaining = 31 then seconds_remaining2 = "52";
	  else if seconds_remaining = 30 then seconds_remaining2 = "50.3";
	  else if seconds_remaining = 29 then seconds_remaining2 = "48.7";
      else if seconds_remaining = 28 then seconds_remaining2 = "47";
	  else if seconds_remaining = 27 then seconds_remaining2 = "45.3";
	  else if seconds_remaining = 26 then seconds_remaining2 = "43.6";
	  else if seconds_remaining = 25 then seconds_remaining2 = "41.9";
	  else if seconds_remaining = 24 then seconds_remaining2 = "40.3";
	  else if seconds_remaining = 23 then seconds_remaining2 = "38.6";
	  else if seconds_remaining = 22 then seconds_remaining2 = "36.9";
	  else if seconds_remaining = 21 then seconds_remaining2 = "35.2";
	  else if seconds_remaining = 20 then seconds_remaining2 = "33.6";
	  else if seconds_remaining = 19 then seconds_remaining2 = "31.9";
	  else if seconds_remaining = 18 then seconds_remaining2 = "30.2";
	  else if seconds_remaining = 17 then seconds_remaining2 = "28.5";
	  else if seconds_remaining = 16 then seconds_remaining2 = "26.8";
	  else if seconds_remaining = 15 then seconds_remaining2 = "25.2";
	  else if seconds_remaining = 14 then seconds_remaining2 = "23.5";
	  else if seconds_remaining = 13 then seconds_remaining2 = "21.8";
	  else if seconds_remaining = 12 then seconds_remaining2 = "20.1";
	  else if seconds_remaining = 11 then seconds_remaining2 = "18.5";
	  else if seconds_remaining = 10 then seconds_remaining2 = "16.8";
	  else if seconds_remaining = 9 then seconds_remaining2 = "15.1";
      else if seconds_remaining = 8 then seconds_remaining2 = "13.4";
	  else if seconds_remaining = 7 then seconds_remaining2 = "11.7";
	  else if seconds_remaining = 6 then seconds_remaining2 = "10.1";
	  else if seconds_remaining = 5 then seconds_remaining2 = "08.4";
	  else if seconds_remaining = 4 then seconds_remaining2 = "06.7";
	  else if seconds_remaining = 3 then seconds_remaining2 = "05";
	  else if seconds_remaining = 2 then seconds_remaining2 = "03.4";
	  else if seconds_remaining = 1 then seconds_remaining2 = "01.7";
	  else if seconds_remaining = 0 then seconds_remaining2 = "00";
else seconds_remaining2 = seconds_remaining;
run;

/* print data */
proc print data=train2 (obs=10);
run;

/* concatenante data */
data train2;
set train2;
pms_remaining = cat(periods_remaining2, minutes_remaining2, seconds_remaining2);
run;

/* print data */
proc print data=train2 (obs=10);
run;

/* make field numeric (some components contained leading zeroes */
data train2;
set train2;
   n_pms_remaining = input(pms_remaining,8.);
run;

/* drop original non-numeric concetenanted data field */
data train2;
set train2 (drop = pms_remaining); 
run;
data train2;

/* rename new numeric concatenated data field  */
set train2 (rename=( 
'n_pms_remaining'n='pms_remaining'n));
run;

/* print data */
proc print data=train2 (obs=10);
run;





/* check data - histogram */
ods graphics on;
proc univariate data = train2;
var pms_remaining;
histogram;
run; 
ods graphics off;

/* check data - scatter plot */
proc sgplot data=train2;
   scatter x=pms_remaining y=shot_made_flag / group=shot_made_flag;
run;



/* transform data - log transformation on shot distance and time remaining */
data train3;
set train2;
l_shot_distance = log(shot_distance);
l_pms_remaining = log(pms_remaining);
run;


/* check data - scatter plot */
ods graphics on;
proc univariate data = train3 plot;
var l_shot_distance l_pms_remaining;
run;
ods graphics off;




/* correlation analysis */
ods graphics on; 
proc corr data=train2 plots=matrix(histogram);                                                                                                                
var recId 
game_event_id 
game_id 
lat 
loc_x 
loc_y 
lon 
minutes_remaining 
period 
playoffs 
season 
seconds_remaining 
shot_distance 
shot_made_flag 
team_id 
game_date 
shot_id 
attendance 
arena_temp 
avgnoisedb;                                                                                                                    
run; 
ods graphics off;



/* principal component analysis */
ods graphics on;
proc princomp plots=all data=train2 cov out=pca;                                                                                                              
var recId 
game_event_id 
game_id 
lat 
loc_x 
loc_y 
lon 
minutes_remaining 
period 
playoffs 
season 
seconds_remaining 
shot_distance 
shot_made_flag 
team_id 
game_date 
shot_id 
attendance 
arena_temp 
avgnoisedb;
run;
ods graphics off;


/* correlation analysis using train2 data vs shot made flag */
proc corr data=train2 plots=matrix(histogram);                                                                                                              
      var shot_made_flag game_event_id lat loc_y minutes_remaining period seconds_remaining shot_distance attendance arena_temp avgnoisedb;                                                                                                                             
      run;


/* correlation analysis using pricipal components vs shot made flag */
proc corr data=pca plots=matrix(histogram);                                                                                                              
      var shot_made_flag prin1 - prin10;                                                                                                                             
      run;


/* model 1 - GLM select using PCA */
proc glmselect data=pca plots=all seed=3;
model shot_made_flag =prin1-prin10 / selection = stepwise(choose=CV select=CV stop=CV);
run;


/* model 2 - Logistic using PCA */
ods graphics on;
proc logistic data=pca plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') =prin1-prin10 / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_2_results p = Predict;
run;
ods graphics off;


/* model 3 - Logistic using train2 dataset (not PCA) only by distance */
ods graphics on;
proc logistic data=train2 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_3_results p = Predict;
run;
ods graphics off;


/* create data for model 4 - data sets for playoffs and not at playoffs */

data train2_playoffs;
set train2;
where playoffs = 1;
run;

data train2_no_playoffs;
set train2;
where playoffs = 0;
run;


/* model 4A - Logistic using train2 dataset (not PCA) during playoffs */

ods graphics on;
proc logistic data=train2_playoffs plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_4A_results p = Predict;
run;
ods graphics off;


/* model 4B - Logistic using train2 dataset (not PCA) during playoffs */

ods graphics on;
proc logistic data=train2_no_playoffs plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_4B_results p = Predict;
run;
ods graphics off;


/* model 5 - Logistic using train2 dataset (not PCA) during playoffs */
ods graphics on;
proc logistic data=train2 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance playoffs arena_temp game_event_id lat lon / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_5_results p = Predict;
run;
ods graphics off;


/* model 6 - Logistic using train2 dataset (not PCA) for all variables that had corr p < 0.0001 */
ods graphics on;
proc logistic data=train2 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance playoffs period minutes_remaining seconds_remaining attendance arena_temp avgnoisedb / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_6_results p = Predict;
run;
ods graphics off;


/* Model 7 - LDA Model */

proc discrim data=train2 outstat=LDAstat method=normal pool=yes
                list crossvalidate;
      class shot_made_flag;
      priors prop;
      var shot_distance playoffs period minutes_remaining seconds_remaining attendance arena_temp avgnoisedb;
   run;
	








/* Import test data for prediction model */

/* import test data */
proc import datafile="c:\users\allen\documents\smu data science\MSDS 6372 - Applied Statistics\project 2\project2pred.csv"  
          dbms=dlm out=test replace;
     delimiter=',';
     getnames=yes;
run;

/* print data */
proc print data=test (obs=10);
run;

/* investigate data for variable types*/
proc contents data=test;
run;

/* look for missing numeric data - may need to impute */
proc means data = test n nmiss;
  var _numeric_;
run;

/* investigate means of training data */
proc means data = test n nmiss;
run;

/* univariate data analysis */
proc univariate data = test;
var season;
run;

/* note that certain "season" fields are missing */
Data _seasontest;
    set test;
    where missing (season);
run;

/* print "season" data, where "season" is missing */
proc print data=_seasontest (obs=200);
run;

/* add empty predicted response field */
data test2;
set test;
shot_made_flag = .;
;

/* print data */
proc print data=test2 (obs=200);
run;


/* Create TRAIN and TEST fields to distinguish test vs train data. Combine data, predict missing values, create final data set */

data train2b;
set train2;
file = "TRAIN";
run;

proc print data=train2b (obs=10);
run;

data test2b;
set test2;
file = "TEST";
run;

proc print data=test2b (obs=10);
run;




/* make a numeric shot_made_flag variable in test data */

data test2c;
set test2b;
   n_shot_made_flag = input(shot_made_flag,8.);
run;

/* drop original non-numeric shot_made_flag */
data test2c;
set test2c (drop = shot_made_flag); 
run;

/* rename numeric shot_made_flag */
data test2c;
set test2c (rename=( 
'n_shot_made_flag'n='shot_made_flag'n));
run;

/* drop the n_shot_made_flag variable */
data test2c;
set test2c (drop = shot_made_flag); 
run;

/* rename rannum variable to recId */
data test2c;
set test2c (rename=( 
'rannum'n='recId'n));
run;



/* combine data sets */

data test3;
set train2b test2c;
run;

proc print data=test3 (obs=10);
run;

proc contents data=test3;
run;


/* predict response field (shot_made_flag) using desired method */
ods graphics on;
proc logistic data=test3 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance playoffs period minutes_remaining seconds_remaining attendance arena_temp avgnoisedb / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
output out = model_test_results p = Predict;
run;
ods graphics off;


/* check data for completeness */

proc means data = results n nmiss;
  var _numeric_;
run;

proc print data=results (obs=10);
where file = "TEST"; 
run;

proc contents data=results; 
run;

proc means data=results
	N Mean Std Min Q1 Median Q3 Max;
run; 

/* This is the final step that maps the predicted value into the shot_made_flag variable
and then drops all variables except shot_id and shot_made_flag. */

data results_final;
retain shot_id shot_made_flag;
set model_test_results;
if shot_made_flag < 1 then shot_made_flag = predict;
keep shot_id shot_made_flag;
where file = "TEST"; 
run;

proc print data=results_final (obs=100);
run;

proc contents data=results_final; 
run;












ods graphics on;
proc logistic data=test3 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_test3_results p = Predict;
run;
ods graphics off;



/* model 5 - Logistic using train2 dataset (not PCA) during playoffs */
ods graphics on;
proc logistic data=test3 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance playoffs arena_temp game_event_id lat lon / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_test5_results p = Predict;
run;
ods graphics off;


/* model 6 - Logistic using train2 dataset (not PCA) for all variables that had corr p < 0.0001 */
ods graphics on;
proc logistic data=test3 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance playoffs period minutes_remaining seconds_remaining attendance arena_temp avgnoisedb / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare
									lackfit
									ctable;
  output out = model_test6_results p = Predict;
run;
ods graphics off;


data results_final_6;
retain shot_id shot_made_flag;
set model_test6_results;
if shot_made_flag < 1 then shot_made_flag = predict;
keep shot_id shot_made_flag;
where file = "TEST"; 
run;

