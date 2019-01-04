clear
%can either be your data as a sub * cond matrix, 
% or name of an excel/csv file as str
prefs.data = 'Exp1_Data.xlsx';

%interval of N to simulate (e.g, 10-100 by 10)
prefs.N_range = 10:10:100; 

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
prefs.comps = [1, 2];

%Run Power Analysis with these settings
pow_results = PowerAnalysis(prefs);