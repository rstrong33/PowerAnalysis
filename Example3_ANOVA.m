clear
%can either be your data as a sub * cond matrix, 
% or name of an excel/csv file as str
prefs.csv_file = 'Data_ANOVA_Mixed.csv';

%interval of N to simulate (e.g, 50-300 by 25)
prefs.N_range = 50:25:200; 

%interval of trials per condition to simulate (e.g, 8-20 by 4)
prefs.trial_range = 8:8:32;

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
prefs.comps = [2, 1];

%does the first main effect need to be significant to be a "success"
%note that for mixed-factors designs, between-subjects factor will always 
%be considered "factor 1"
prefs.sig_ME1 = false;

%does the second main effect need to be significant to be a "success"
%note that for mixed-factors designs, within-subjects factor will always 
%be considered "factor 2"
prefs.sig_ME2 = false;

%does the interaction need to be significant to be a "success"
prefs.sig_int = true;

%FOR BETWEEN-SUBJECTS OR MIXED DESIGNS ONLY (ignored otherwise)
%how participants should be split between between-factor levels
%needs a value for each between-subjects factor level, and sum to 1 (100%)
%for example, if 60 participants in 2 condition between-subjects design,
%prefs.condition_allocation = [.5, .5] would have 30 subs/condition.
%[.75, .25] would result in condition 1 = 45 subs, condition 2 = 15 subs
%[1/3, 2/3] would result in condition 1 = 20 subs, condition 2 = 40 subs
prefs.condition_allocation = [.5, .5];

%Run Power Analysis with these settings
pow_results = PowerAnalysis_ANOVA(prefs);