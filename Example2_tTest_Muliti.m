clear
%can either be your data as a sub * cond matrix, 
% or name of an excel/csv file as str
prefs.csv_file = 'Data_tTest_Within_Multi.csv';

%interval of N to simulate (e.g, 10-100 by 10)
%for between-subjects designs, this is TOTAL subjects (not per condition)
prefs.N_range = 100:50:300;

%interval of trials per condition to simulate (e.g, 8-24 by 4)
prefs.trial_range = 8:4:24;

%p value to use in statisical test during simulation
prefs.alpha = .05;

%number of experiments to simulate per trial*N combination
%higher number of sims will give more stable/accurate power estimates, 
%but will be slower. 10000 or 100000 is usually good
prefs.nSims = 10000;

%what comparisons do you want to make? Should be a comparison * 2 vector,
%with condition that should be larger on the left
%for example, if you expect condition 1 to be larger than condition 2, you
%should enter [1, 2];
%when you run this script, a graph will display how your conditions have
%been numbered (adjust below and run again if necessary)
prefs.comps = [4, 1
    4, 2
    4, 3
    2, 1
    3, 1];

%FOR BETWEEN-SUBJECTS DESIGNS ONLY (ignored otherwise)
%how participants should be split between conditions, must sum to 1 (100%)
%for example, if 60 participants in 2 condition between-subjects design,
%prefs.condition_allocation = [.5, .5] would have 30 subs/condition.
%[.75, .25] would result in condition 1 = 45 subs, condition 2 = 15 subs
%[1/3, 2/3] would result in condition 1 = 20 subs, condition 2 = 40 subs
prefs.condition_allocation = [];

%Run Power Analysis with these settings
power_results = PowerAnalysis_tTests(prefs);