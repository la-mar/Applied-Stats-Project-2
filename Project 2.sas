/* import data */

proc import datafile="c:\users\allen\documents\smu data science\MSDS 6372 - Applied Statistics\project 2\project2Data.csv"  
          dbms=dlm out=train replace;
     delimiter=',';
     getnames=yes;
run;

proc print data=train (obs=10);
run;

proc contents data=train;
run;

proc means data = train n nmiss;
  var _numeric_;
run;

proc means data = train n nmiss;
run;

proc univariate data = train;
var season;
run;

Data _season;
    set train;
    where missing (season);
run;

proc print data=_season (obs=200);
run;


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

proc print data=train2 (obs=10);
run;

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

proc print data=train2 (obs=10);
run;

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

proc print data=train2 (obs=10);
run;

data train2;
set train2;
pms_remaining = cat(periods_remaining2, minutes_remaining2, seconds_remaining2);
run;

proc print data=train2 (obs=10);
run;

data train2;
set train2;
   n_pms_remaining = input(pms_remaining,8.);
run;
data train2;
set train2 (drop = pms_remaining); 
run;
data train2;
set train2 (rename=( 
'n_pms_remaining'n='pms_remaining'n));
run;
proc print data=train2 (obs=10);
run;






ods graphics on;
proc univariate data = train2;
var pms_remaining;
histogram;
run; 
ods graphics off;


proc sgplot data=train2;
   scatter x=pms_remaining y=shot_made_flag / group=shot_made_flag;
run;




data train3;
set train2;
l_shot_distance = log(shot_distance);
l_pms_remaining = log(pms_remaining);
run;

ods graphics on;
proc univariate data = train3 plot;
var l_shot_distance l_pms_remaining;
run;
ods graphics off;




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

proc corr data=pca plots=matrix(histogram);                                                                                                              
      var shot_made_flag prin1 - prin10;                                                                                                                             
      run;


proc glmselect data=pca plots=all seed=3;
model shot_made_flag=prin1-prin10 / selection = stepwise(choose=CV select=CV stop=CV);
run;

/*11-5-18 */

ods graphics on;
proc logistic data=pca plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') =prin1-prin10 / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare;
run;
ods graphics off;


ods graphics on;
proc logistic data=train2 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare;
run;
ods graphics off;

ods graphics on;
proc logistic data=train2 plots(only)=(roc(id=obs) effect);
  model shot_made_flag (event='1') = shot_distance playoffs / scale=none
                            		clparm=wald
                            		clodds=pl
                            		rsquare;
run;
ods graphics off;
