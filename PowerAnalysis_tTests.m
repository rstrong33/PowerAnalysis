function power_results = PowerAnalysis_tTests(prefs)

%maximum array elements during simulation
prefs.max_array_size = 1000000;
prefs.figure_width = 700;
prefs.figure_height = 700;

%load csv
csvdata = textscan(fopen(prefs.csv_file), '%s%s%s', 'delimiter', ',');

%organize data
header = {csvdata{1}{1}, csvdata{2}{1}, csvdata{3}{1}};
sub_list = csvdata{1}(2:end);
ans_list = csvdata{2}(2:end);
cond_list = csvdata{3}(2:end);
ans_as_nums = str2num(cell2mat(ans_list));

%convert subject names to integers
sub_names = unique(csvdata{1}(2:end), 'stable');
sub_nums = zeros(length(csvdata{1}(2:end)), 1);
for sn = 1:length(sub_names)
    sub_nums(strcmp(sub_list, sub_names(sn)), 1) = sn;
end

%convert condition names to integers
cond_names = unique(csvdata{3}(2:end));
cond_nums = zeros(length(csvdata{3}(2:end)), 1);
for cn = 1:length(cond_names)
    cond_nums(strcmp(cond_list, cond_names(cn)), 1) = cn;
end

%subs, scores, and conditions as numbers
prefs.data = [sub_nums, ans_as_nums, cond_nums];

%condition names (original as text)
prefs.cond_names = cond_names;

%column headers from CSV data
prefs.header = header;

%determine whether a within-subject or between-subject design
subs = unique(prefs.data(:,1));
if length(unique(prefs.data(prefs.data(:,1) == subs(1),3))) == 1
    prefs.within_between = 2; %between subjects design
else
    prefs.within_between = 1; %within subjects design
end

%determine whether DV is percent correct (only 2 choices)
if length(unique(prefs.data(:,2))) == 2
    
    tMax = max(prefs.trial_range);
    nMax = max(prefs.N_range);
    nConds = length(cond_names);
    
    if prefs.within_between == 1  %within-subjects design
        try
            zeros(nMax, (tMax+1)^nConds);
            fast_sim = true;
        catch
            fast_sim = false;
        end
    elseif prefs.within_between == 2  %between-subjects design design
        fast_sim = true;
    end
else
    fast_sim = false;
end

%make pilot data graph
MakePilotGraph(prefs);

%main power analysis
if prefs.within_between == 1
    if fast_sim
        power_results = PowerAnalysisWithinFast(prefs);
    else
        power_results = PowerAnalysisWithin(prefs);
    end
elseif prefs.within_between == 2
    if fast_sim
        power_results = PowerAnalysisBetweenFast(prefs);
    else
        power_results = PowerAnalysisBetween(prefs);
    end
end

end

function power_results = PowerAnalysisWithin(prefs)

%prefs.data = single(prefs.data);

%simulation info
nSims = prefs.nSims; %number of experiments to simulate
nPilotSubs = length(unique(prefs.data(:,1))); %how many subjects in actual data
sub_names = unique(prefs.data(:,1));
sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest
nConds = length(unique(prefs.data(:,3))); %number of conditions
cond_names = unique(prefs.data(:,3));
nPilotTrials = sum(prefs.data(:,1) == sub_names(1) & prefs.data(:,3) == cond_names(1));

%organize data sub*trial*cond
pilot_data = zeros(nPilotSubs, nPilotTrials, nConds);
for s = 1:nPilotSubs
    for cond = 1:nConds
        pilot_data(s,:,cond) = prefs.data(prefs.data(:,1) == s & prefs.data(:,3) == cond, 2)';
    end
end

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));
dz_vect = cell(1, nComps);

for trial_count= 1:length(trial_vector)
    
    t = trial_vector(trial_count);
    
    %figure out sampling variability based upon number of subjects
    for sub_count = 1:length(sub_vector)
        
        clc
        disp([num2str(round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)))), '% Complete']);
        
        %number of subjects to simulate
        nSubs = sub_vector(sub_count);
        sample_size(trial_count, sub_count) = nSubs;
        num_trials(trial_count, sub_count) = t;
        
        total_num_trials = nSubs*nSims*t;
        sim_ratio = total_num_trials/prefs.max_array_size;
        total_sim_rounds = ceil(sim_ratio);
        sims_per_round = floor(nSims/sim_ratio);
        sims_last_round = nSims - (total_sim_rounds-1)*sims_per_round;
        simsPerRound = [repmat(sims_per_round, 1, total_sim_rounds - 1), sims_last_round];
        
        if simsPerRound(end) > sims_per_round
            nLeft = simsPerRound(end);
            simsPerRound(end) = [];
            total_sim_rounds = total_sim_rounds - 1;
            xRatio = nLeft/sims_per_round;
            for c = 1:floor(xRatio)
                simsPerRound(end+1) = sims_per_round;
                total_sim_rounds = total_sim_rounds + 1;
            end
            if xRatio ~= floor(xRatio)
                simsPerRound(end+1) = nLeft - floor(xRatio)*sims_per_round;
                total_sim_rounds = total_sim_rounds + 1;
            end
        end
        
        for c = 1:nConds
            cond_scores{c} = [];
        end
        
        for sim_round = 1:total_sim_rounds
            
            nSimsRound = simsPerRound(sim_round);
            
            %select random subjects
            sim_subs = randsample(1:nPilotSubs, nSubs*nSimsRound, 'true');
            sim_subs = repelem(sim_subs, t)';
            
            %select random trials, for each condition, generate data
            trial_nums = zeros(length(sim_subs), nConds);
            for c = 1:nConds
                trial_nums(:,c) = ceil(rand(length(sim_subs), 1)*nPilotTrials);
                cond_scores_tmp = pilot_data(sub2ind(size(pilot_data), sim_subs, trial_nums(:,c), repmat(c, length(sim_subs), 1)));
                cond_scores{c} = [cond_scores{c}, squeeze(mean(permute(reshape(cond_scores_tmp, t, nSubs, nSimsRound), [2 1 3]),2))];
            end
        end
        
        % do condition comparisons
        %p = cell(1, nComps);
        power_marker = zeros(nComps, nSims);
        for comp = 1:nComps
            diff_scores = cond_scores{prefs.comps(comp,1)} - cond_scores{prefs.comps(comp,2)};
            dz_vect{trial_count, sub_count}{comp} = mean(diff_scores)./std(diff_scores);
            [~,p] = ttest(diff_scores);
            power_marker(comp, :) = p < prefs.alpha & dz_vect{trial_count, sub_count}{comp} > 0;
        end
        
        power(trial_count, sub_count) = mean(all(power_marker, 1));
    end
end

clc
disp('100% Complete')

%output information
power_results.power = power; %power for each simulated design
power_results.n = sample_size; %sample size for each simulated design
power_results.num_trials = num_trials; %number of trials for each design
power_results.dz_vect = dz_vect; %effect size vector for each design
power_results.sub_vector = sub_vector;
power_results.trial_vector = trial_vector;
power_results.prefs.figure_width = prefs.figure_width;
power_results.prefs.figure_height = prefs.figure_height;

PlotHeatMap(power_results);

end

function power_results = PowerAnalysisBetween(prefs)

if sum(prefs.condition_allocation) ~= 1
    error('Condition allocation does not add up to 1 (100%). Use fractions if this is due to rounding error (e.g., use 1/3 instead of .33)')
elseif length(prefs.condition_allocation) ~= length(prefs.cond_names)
    error('Must have an allocation amount for each between-subjects level.')
end

%simulation info
nSims = prefs.nSims; %number of experiments to simulate
nConds = length(unique(prefs.data(:,3)));
sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest
cond_names = unique(prefs.data(:,3));

nPilotSubs = length(unique(prefs.data(:,1)));
for c = 1:nConds
   cond_subs{c} = unique(prefs.data(prefs.data(:,3) == c,1));
   nPilotSubs(c) = length(cond_subs{c}); %how many subjects in actual data per condition 
   nPilotTrials(c) = sum(prefs.data(:,1) == cond_subs{c}(1) & prefs.data(:,3) == cond_names(c));
end

%organize data sub*trial, for each cond
for cond = 1:nConds
    for s = 1:nPilotSubs(cond)
        pilot_data{cond}(s,:) = prefs.data(prefs.data(:,1) == cond_subs{cond}(s) & prefs.data(:,3) == cond, 2)';
    end
end

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
subs_by_cond = cell(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));
ds_vect = cell(1, nComps);


for trial_count= 1:length(trial_vector)
    
    t = trial_vector(trial_count);
    
    %figure out sampling variability based upon number of subjects
    for sub_count = 1:length(sub_vector)
        
        clc
        disp([num2str(round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)))), '% Complete']);
        
        %number of subjects to simulate per condition
        nSubs_Total = sub_vector(sub_count);
        nSubs = zeros(1,nConds);

        for c = 1:nConds
            nSubs(c) = round(nSubs_Total*prefs.condition_allocation(c));
            cond_scores{c} = [];
        end

        subs_by_cond{trial_count, sub_count} = nSubs;
        sample_size(trial_count, sub_count) = nSubs_Total;
        num_trials(trial_count, sub_count) = t;
        
        
        for c = 1:nConds
            total_num_trials = nSubs(c)*nSims*t;
            sim_ratio = total_num_trials/prefs.max_array_size;
            total_sim_rounds = ceil(sim_ratio);
            sims_per_round = floor(nSims/sim_ratio);
            sims_last_round = nSims - (total_sim_rounds-1)*sims_per_round;
            simsPerRound = [repmat(sims_per_round, 1, total_sim_rounds - 1), sims_last_round];
            
            if simsPerRound(end) > sims_per_round
                nLeft = simsPerRound(end);
                simsPerRound(end) = [];
                total_sim_rounds = total_sim_rounds - 1;
                xRatio = nLeft/sims_per_round;
                for c1 = 1:floor(xRatio)
                    simsPerRound(end+1) = sims_per_round;
                    total_sim_rounds = total_sim_rounds + 1;
                end
                if xRatio ~= floor(xRatio)
                    simsPerRound(end+1) = nLeft - floor(xRatio)*sims_per_round;
                    total_sim_rounds = total_sim_rounds + 1;
                end
            end
            
            for sim_round = 1:total_sim_rounds
                
                nSimsRound = simsPerRound(sim_round);
                
                %select random subjects
                sim_subs = randsample(1:nPilotSubs(c), nSubs(c)*nSimsRound, 'true');
                sim_subs = repelem(sim_subs, t)';
                
                trial_nums = ceil(rand(length(sim_subs), 1)*nPilotTrials(c));
   
                cond_scores_tmp = pilot_data{c}(sub2ind(size(pilot_data{c}), sim_subs, trial_nums));

                cond_scores{c} = [cond_scores{c}, squeeze(mean(permute(reshape(cond_scores_tmp, t, nSubs(c), nSimsRound), [2 1 3]),2))];         
            end
        end
        % do condition comparisons
        power_marker = zeros(nComps, nSims);
        for comp = 1:nComps
            c1 = cond_scores{prefs.comps(comp,1)};
            c2 = cond_scores{prefs.comps(comp,2)};
            
            ds_vect{trial_count, sub_count}{comp} = (mean(c1) - mean(c2)) ./...
                (((nSubs(prefs.comps(comp,1)) - 1)*(std(c1).^2) + (nSubs(prefs.comps(comp,2)) - 1)*(std(c2).^2))/(nSubs(prefs.comps(comp,1)) + nSubs(prefs.comps(comp,2)) - 2)).^.5;
            [~,p] = ttest2(c1,c2);
            

            power_marker(comp, :) = p < prefs.alpha & ds_vect{trial_count, sub_count}{comp} > 0;
        end
        
        power(trial_count, sub_count) = mean(all(power_marker, 1));
    end
end

clc
disp('100% Complete')

%output information
power_results.power = power; %power for each simulated design
power_results.n = sample_size; %sample size for each simulated design
power_results.num_trials = num_trials; %number of trials for each design
power_results.ds_vect = ds_vect; %effect size vector for each design
power_results.sub_by_cond = subs_by_cond;  %subs in each cond for each design
power_results.sub_vector = sub_vector;
power_results.trial_vector = trial_vector;
power_results.prefs.figure_width = prefs.figure_width;
power_results.prefs.figure_height = prefs.figure_height;

PlotHeatMap(power_results);
end

function power_results = PowerAnalysisWithinFast(prefs)

%set up data (mean scores, nSubs * nConds)
subs = unique(prefs.data(:,1));
nConds = length(unique(prefs.data(:,3)));
nPilotSubs = length(subs); %how many subjects in actual data

for s = 1:nPilotSubs
    for c = 1:nConds
        pilot_data(s, c) = mean(prefs.data(prefs.data(:,1) == subs(s) & ...
            prefs.data(:,3) == c,2));
    end
end

%simulation info
nSims = prefs.nSims; %number of experiments to simulate
sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));
dz_vect = cell(1, nComps);

for trial_count= 1:length(trial_vector)
    
    t = trial_vector(trial_count);
    
    %determine condition difference pdf for each subject in pilot data
    cFinal = zeros(nPilotSubs, (t+1)^nConds);
    outcomes = 1:(t+1)^nConds;
    
    %calculate measurement variability for each subject based upon 
    %condition means and number of trials
    for n = 1:nPilotSubs
        
        cond_probs = cell(1, nConds);
        for c = 1:nConds
            cond_probs{c} = binopdf(0:t, t, pilot_data(n,c));
        end
        
        %first two conditions
        tmp =  cond_probs{1}' * cond_probs{2};
        tmp = reshape(tmp, numel(tmp), 1);
        
        %additional conditions
        if nConds > 2
            for c = 3:nConds
                tmp = reshape(tmp*cond_probs{c}, numel(tmp)*length(cond_probs{c}), 1);
            end
        end
        
        cond_score = cell(1, nConds);
        for c = 1:nConds
           cond_score{c} = repmat(repelem(0:t, (t+1)^(c-1))', (t+1)^(nConds-c), 1);
        end
        
        cFinal(n, :) = tmp;
    end
    
    %figure out sampling variability based upon number of subjects
    for sub_count = 1:length(sub_vector)
        
        clc
        disp([num2str(round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)))), '% Complete']);
        
        %number of subjects to simulate
        nSubs = sub_vector(sub_count);
        outcome_samples = randsample(outcomes, nSims*nSubs, 'true', mean(cFinal));
        sample_size(trial_count, sub_count) = nSubs;
        num_trials(trial_count, sub_count) = t;
        
        % do condition comparisons
        %p = cell(1, nComps);
        power_marker = zeros(nComps, nSims);
        for comp = 1:nComps
            diff_scores = cond_score{prefs.comps(comp,1)}(outcome_samples) - cond_score{prefs.comps(comp,2)}(outcome_samples);
            diff_scores = reshape(diff_scores, nSubs, nSims);
            dz_vect{comp}{trial_count, sub_count} = mean(diff_scores)./std(diff_scores);
            [~,p] = ttest(diff_scores);
            power_marker(comp, :) = p < prefs.alpha & dz_vect{comp}{trial_count, sub_count} > 0;
        end
        
        power(trial_count, sub_count) = mean(all(power_marker, 1));
    end
end

clc
disp('100% Complete')

%output information
power_results.power = power; %power for each simulated design
power_results.n = sample_size; %sample size for each simulated design
power_results.num_trials = num_trials; %number of trials for each design
power_results.dz_vect = dz_vect; %effect size vector for each design
power_results.sub_vector = sub_vector;
power_results.trial_vector = trial_vector;
power_results.prefs.figure_width = prefs.figure_width;
power_results.prefs.figure_height = prefs.figure_height;

PlotHeatMap(power_results);

end

function power_results = PowerAnalysisBetweenFast(prefs)

if sum(prefs.condition_allocation) ~= 1
    error('Condition allocation does not add up to 1 (100%). Use fractions if this is due to rounding error (e.g., use 1/3 instead of .33)')
elseif length(prefs.condition_allocation) ~= length(prefs.cond_names)
    error('Must have an allocation amount for each between-subjects level.')
end

%Get data setup properly (array slot for each condition, 
%simulation info
nSims = prefs.nSims; %number of experiments to simulate
nConds = length(unique(prefs.data(:,3)));
sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest

nPilotSubs = length(unique(prefs.data(:,1)));
for c = 1:nConds
   cond_subs{c} = unique(prefs.data(prefs.data(:,3) == c,1));
   nPilotSubs(c) = length(cond_subs{c}); %how many subjects in actual data per condition
   for s = 1:nPilotSubs(c)
      pilot_data{c}(s) = mean(prefs.data(prefs.data(:,1) == cond_subs{c}(s) & ...
            prefs.data(:,3) == c,2));
   end
end

%

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
subs_by_cond = cell(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));
ds_vect = cell(1, nComps);


for trial_count= 1:length(trial_vector)
    
    t = trial_vector(trial_count);
    
    %determine condition difference pdf for each subject in pilot data
    cFinal = cell(1, nConds);
    outcomes = 0:t;
    
    %calculate measurement variability for each subject based upon 
    %condition means and number of trials
    
    cond_prob = cell(1, nConds);
    for c = 1:nConds
        for n = 1:nPilotSubs(c)
            cond_prob{c}(n,:) = binopdf(0:t, t, pilot_data{c}(n));
        end
        cFinal{c} = mean(cond_prob{c});
    end
    
    
    %figure out sampling variability based upon number of subjects
    for sub_count = 1:length(sub_vector)
        
        clc
        disp([num2str(round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)))), '% Complete']);
        
        %number of subjects to simulate per condition
        
        nSubs_Total = sub_vector(sub_count);
        outcome_samples = cell(1, nConds);
        nSubs = zeros(1,nConds);

        for c = 1:nConds
            nSubs(c) = round(nSubs_Total*prefs.condition_allocation(c));
            outcome_samples{c} = randsample(outcomes, nSims*nSubs(c), 'true', cFinal{c});
        end
        
        subs_by_cond{trial_count, sub_count} = nSubs;
        sample_size(trial_count, sub_count) = nSubs_Total;
        num_trials(trial_count, sub_count) = t;
        
        
        % do condition comparisons
        %p = cell(1, nComps);
        power_marker = zeros(nComps, nSims);
        for comp = 1:nComps
            c1 = reshape(outcome_samples{prefs.comps(comp,1)}, nSubs(prefs.comps(comp,1)), nSims);
            c2 = reshape(outcome_samples{prefs.comps(comp,2)}, nSubs(prefs.comps(comp,2)), nSims);
            ds_vect{comp}{trial_count, sub_count} = (mean(c1) - mean(c2)) ./...
                (((nSubs(prefs.comps(comp,1)) - 1)*(std(c1).^2) + (nSubs(prefs.comps(comp,2)) - 1)*(std(c2).^2))/(nSubs(prefs.comps(comp,1)) + nSubs(prefs.comps(comp,2)) - 2)).^.5;
            [~,p] = ttest2(c1,c2);
            power_marker(comp, :) = p < prefs.alpha & ds_vect{comp}{trial_count, sub_count} > 0;
            
        end
        
        power(trial_count, sub_count) = mean(all(power_marker, 1));
    end
end

clc
disp('100% Complete')

%output information
power_results.power = power; %power for each simulated design
power_results.n = sample_size; %sample size for each simulated design
power_results.num_trials = num_trials; %number of trials for each design
power_results.ds_vect = ds_vect; %effect size vector for each design
power_results.sub_by_cond = subs_by_cond;  %subs in each cond for each design
power_results.sub_vector = sub_vector;
power_results.trial_vector = trial_vector;
power_results.prefs.figure_width = prefs.figure_width;
power_results.prefs.figure_height = prefs.figure_height;

PlotHeatMap(power_results);
end

function MakePilotGraph(prefs)
%condition number labels
bar_color = [.75,.75,.75];

conds = unique(prefs.data(:,3));
nConds = length(conds);

for c = 1:nConds
    condition_means(c) = mean(prefs.data(prefs.data(:,3) == c, 2));
end

figure(1)
clf
hold on
subplot(2,1,1)
hold on
title('Pilot Data')
bar(1:nConds, condition_means, 'FaceColor', bar_color)

range = max(condition_means) - min(condition_means);
if range == 0
   range = max(condition_means);
end
drop = .5;
ymin = min(condition_means) - drop*range;
ymax = max(condition_means) + drop*range;
ylim([ymin, ymax])
label_y = min(condition_means) - (drop/2)*range;
xlim([0, nConds + 1])


range = 0:(nConds + 1);
for c = 1:length(range)
    x_label{c} = '';
end
for c = 1:nConds
    x_label{c+1} = prefs.cond_names{c};
end
set(gca,'XTick',range)
set(gca, 'XTickLabel',x_label, 'fontsize',24)


ylabel(prefs.header{2})

%bar labels
fs = 16;
for b = 1:length(condition_means)
    text(b, label_y, num2str(b), 'FontSize', fs);
end

subplot(2,1,2)
title('Power Simulation Parameters', 'FontSize', 24)
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'XColor','none')
set(gca,'YColor','none')
xlim([0,1])
ylim([0,1])
spot = .9;
spot_jump = .1;
text(.1, .9, 'Successful Study Requies:', 'FontSize', 16)
count = 0;

%t-tests
for p = 1:size(prefs.comps, 1)
    count = count + 1;
    spot = spot - spot_jump;
    if prefs.within_between == 1
        txt = [num2str(count), ': ', num2str(prefs.comps(p, 1)), ' > ', num2str(prefs.comps(p, 2)), ' (within subjects)'];
    else
        txt = [num2str(count), ': ', num2str(prefs.comps(p, 1)), ' > ', num2str(prefs.comps(p, 2)), ' (between subjects)'];
    end
    text(.2, spot, txt, 'FontSize', 16);
end

%position figure on screen
set(0,'units','pixels')
Pix_SS = get(0,'screensize');
w = prefs.figure_width;
h = prefs.figure_height;
set(gcf, 'Position',  [Pix_SS(3)-1.75*w, Pix_SS(4) - h, w, h])
end

function PlotHeatMap(power_results)
%plot
figure(2)
clf

power = power_results.power;
power = round(power, 2);

if all(all(power == 0))
    xlim([0,1])
    ylim([0,1])
    text(.27, .5,'Cannot Display Heatmap', 'FontSize', 20)
    text(.2, .4,'Power for all study designs is 0%', 'FontSize', 20)
    %position figure on screen
    set(0,'units','pixels')
    Pix_SS = get(0,'screensize');
    w = power_results.prefs.figure_width;
    h = power_results.prefs.figure_height;
    set(gcf, 'Position',  [Pix_SS(3)-w*.75, Pix_SS(4) - h*.5, w*.75, .6*h])
elseif numel(power) == 1
    xlim([0,1])
    ylim([0,1])
    text(.2, .6,['Total # of Subjects = ', num2str(power_results.n)], 'FontSize', 20)
    text(.2, .5,['# of Trials Per Condition = ', num2str(power_results.num_trials)], 'FontSize', 20)
    text(.2, .4,['Design Power = ', num2str(power)], 'FontSize', 20)
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'XColor','none')
    set(gca,'YColor','none')
    %position figure on screen
    set(0,'units','pixels')
    Pix_SS = get(0,'screensize');
    w = power_results.prefs.figure_width;
    h = power_results.prefs.figure_height;
    set(gcf, 'Position',  [Pix_SS(3)-w*.75, Pix_SS(4) - h*.5, w*.75, .6*h])
else
    %position figure on screen
    set(0,'units','pixels')
    Pix_SS = get(0,'screensize');
    w = power_results.prefs.figure_width;
    h = power_results.prefs.figure_height;
    set(gcf, 'Position',  [Pix_SS(3)-w*.75, Pix_SS(4) - h*.5, w*.75, .6*h])
    heatmap(power, power_results.sub_vector, power_results.trial_vector, true, 'GridLines', '-', 'FontSize', 14);
    xlabel('Total # of Subjects', 'FontSize', 20)
    ylabel('# of Trials Per Condition', 'FontSize', 20)
    title('Power by N and # of Trials', 'FontSize', 20)
end
end

%plotting functions

function [hImage, hText, hXText] = heatmap(mat, xlab, ylab, textmat, varargin)
% HEATMAP displays a matrix as a heatmap image
%
% USAGE:
% [hImage, hText, hTick] = heatmap(matrix, xlabels, ylabels, textmatrix, 'param', value, ...)
%
% INPUTS:
% * HEATMAP displays "matrix" as an image whose color intensities reflect
%   the magnitude of the values in "matrix". 
%
% * "xlabels" (and "ylabels") can be either a numeric vector or cell array
%   of strings that represent the columns (or rows) of the matrix. If either
%   is not specified or empty, no labels will be drawn. 
%
% * "textmat" can either be: 1 (or true), in which case the "matrix" values will be
%   displayed in each square, a format string, in which case the matrix
%   values will be displayed formatted according to the string specified, a numeric
%   matrix the size of "matrix", in which case those values will be displayed as
%   strings or a cell matrix of strings the size of "matrix", in which case each
%   string will be displayed. If not specified or empty, no text will be
%   displayed on the image
%
% OTHER PARAMETERS (passed as parameter-value pairs)
% * 'Colormap': Either a matrix of size numLevels-by-3 representing the
%   colormap to be used or a string or function handle representing a
%   function that returns a colormap, example, 'jet', 'hsv' or @cool.
%   Non-standard colormaps available within HEATMAP include 'money' and 'red'.
%   By default, the current figure's colormap is used.
%
% * 'ColorLevels': The number of distinct levels in the colormap (default:
%   64). If more levels are specified than are present in the colormap, the
%   levels in the colormap are interpolated. If fewer are specified the
%   colormap is downsampled.
%
% * 'UseLogColormap': A true/false value which, if true, specifies that the
%   intensities displayed should match the log of the "matrix" values. Use
%   this if the data is naturally on a logarithmic scale (default: false)
%
% * 'UseFigureColormap': Specifies whether the figure's colormap should be
%   used. If false, the color intensities after applying the
%   specified/default colormap will be hardcoded, so that the image will be
%   independent of the figure's colormap. If this option is true, the figure
%   colormap in the end will be replaced by specified/default colormap.
%   (default = true)
%
% * 'NaNColor': A 3-element [R G B] vector specifying the color used to display NaN
%   or missing value. [0 0 0] corresponds to black and [1 1 1] to white. By
%   default MATLAB displays NaN values using the color assigned to the
%   lowest value in the colormap. Specifying this option automatically sets
%   the 'UseFigureColormap' option to false because the color mapping must
%   be computed prior to setting the nan color. 
%
% * 'MinColorValue': A scalar number corresponding to the value of the data
%   that is mapped to the lowest color of the colormap. By default this is 
%   the minimum value of the matrix input. 
%
% * 'MaxColorValue': A scalar number corresponding to the value of the data
%   that is mapped to the highest color of the colormap. By default this is 
%   the maximum value of the matrix input. 
% 
% * 'Parent': Handle to an axes object
%
% * 'TextColor': Either a color specification of all the text displayed on
%   the image or a string 'xor' which sets the EraseMode property of the text
%   objects to 'xor'. This will display all the text labels in a color that
%   contrasts its background.
%
% * 'FontSize': The initial fontSize of the text labels on the image. As
%   the image size is scaled the fontSize is shrunk appropriately.
%
% * 'ColorBar': Display colorbar. The corresponding value parameter should
%   be either logical 1 or 0 or a cell array of any additional parameters
%   you wish to pass to the colorbar function (such as location)
%
% * 'GridLines': Draw grid lines separating adjacent sections of the
%   heatmap. The value of the parameter is a LineStyle specification, for example,
%   :, -, -. or --. By default, no grid lines are drawn.
%
% * 'TickAngle': Angle of rotation of tick labels on x-axis. (Default: 0)
%
% * 'ShowAllTicks': Set to 1 or true to force all ticks and labels to be
%   drawn. This can make the axes labels look crowded. (Default: false)
%
% * 'TickFontSize': Font size of the X and Y tick labels. Default value is
%   the default axes font size, usually 10. Set to a lower value if many
%   tick labels are being displayed
%
% * 'TickTexInterpreter': Set to 1 or true to render tick labels using a TEX
%   interpreter. For example, '_b' and '^o' would be rendered as subscript
%   b and the degree symbol with the TEX interpreter. This parameter is only
%   available in MATLAB R2014b and above (Default: false)
%
% OUTPUTS:
% * hImage: handle to the image object
% * hText : handle to the text objects (empty if no text labels are drawn)
% * hTick : handle to the X-tick label text objects if tick angle is not 0
%           (empty otherwise)
%
% Notes:
% * The 'money' colormap displays a colormap where 0 values are mapped to
%   white, negative values displayed in varying shades of red and positive
%   values in varying shades of green
% * The 'red' colormap maps 0 values to white and higher values to red
%
% EXAMPLES:
% data = reshape(sort(randi(100,10)),10,10)-50;
% heatmap(data, cellstr(('A':'J')'), mean(data,2), '%0.0f%%',...
%         'Colormap', 'money', 'Colorbar', true, 'GridLines', ':',...
%         'TextColor', 'b')
% For detailed examples, see the associated document heatmap_examples.m

% Copyright The MathWorks, Inc. 2009-2014

% Handle missing inputs
if nargin < 1, error('Heatmap requires at least one input argument'); end
if nargin < 2, xlab = []; end
if nargin < 3, ylab = []; end
if nargin < 4, textmat = []; end

% Parse parameter/value inputs
p = parseInputs(mat, varargin{:});

% Get heatmap axes information if it already exists
p.axesInfo = getHeatmapAxesInfo(p.hAxes);

% Calculate the colormap based on inputs
p = calculateColormap(p, mat);

% Create heatmap image
p = plotHeatmap(p, mat); % New properties hImage and cdata added

% Generate grid lines if selected
generateGridLines(p);

% Set axes labels
[p, xlab, ylab, hXText, origPos] = setAxesTickLabels(p, xlab, ylab);

% Set text labels
[p, displayText, fontScaleFactor] = setTextLabels(p, mat, textmat);

% Add colorbar if selected
addColorbar(p, mat, textmat)

% Store heatmap properties in axes for callbacks
axesInfo = struct('Type', 'heatmap', 'Parameters', p, 'FontScaleFactor', ...
                   fontScaleFactor, 'mat', mat, 'hXText', hXText, ...
                   'origAxesPos', origPos);
axesInfo.xlab = xlab;
axesInfo.ylab = ylab;
axesInfo.displayText = displayText;
set(p.hAxes, 'UserData', axesInfo);

% Define callbacks
dObj = datacursormode(p.hFig);
set(dObj, 'Updatefcn', @cursorFun);

zObj = zoom(p.hFig);
set(zObj, 'ActionPostCallback', @(obj,evd)updateLabels(evd.Axes,true));

pObj = pan(p.hFig);
% set(pObj, 'ActionPreCallback',  @prePan);
set(pObj, 'ActionPostCallback', @(obj,evd)updateLabels(evd.Axes,true));

set(p.hFig, 'ResizeFcn', @resize)

% Set outputs
hImage = p.hImage;
hText = p.hText;

end

% ---------------------- Heatmap Creation Functions ----------------------

% Parse PV inputs & return structure of parameters
function param = parseInputs(mat, varargin) 

p = inputParser;
p.addParamValue('Colormap',[]); %#ok<*NVREPL>
p.addParamValue('ColorLevels',[]);
p.addParamValue('TextColor',[0 0 0]);
p.addParamValue('UseFigureColormap',true);
p.addParamValue('UseLogColormap',false);
p.addParamValue('Parent',NaN);
p.addParamValue('FontSize',[]);
p.addParamValue('Colorbar',[]);
p.addParamValue('GridLines','none');
p.addParamValue('TickAngle',0);
p.addParamValue('ShowAllTicks',false);
p.addParamValue('TickFontSize',[]);
p.addParamValue('TickTexInterpreter',false);
p.addParamValue('NaNColor', [NaN NaN NaN], @(x)isnumeric(x) && length(x)==3 && all(x>=0) && all(x<=1));
p.addParamValue('MinColorValue', nan, @(x)isnumeric(x) && isscalar(x));
p.addParamValue('MaxColorValue', nan, @(x)isnumeric(x) && isscalar(x));
p.parse(varargin{:});

param = p.Results;

if ~ishandle(param.Parent) || ~strcmp(get(param.Parent,'type'), 'axes')
    param.Parent = gca;
end

ind = ~isinf(mat(:)) | isnan(mat(:));
if isnan(param.MinColorValue)
    param.MinColorValue = min(mat(ind));
end
if isnan(param.MaxColorValue)
    param.MaxColorValue = max(mat(ind));
end

% Add a few other parameters
param.hAxes = param.Parent;
param.hFig = ancestor(param.hAxes, 'figure');
param.IsGraphics2 = ~verLessThan('matlab','8.4');
param.ExplicitlyComputeImage = ~all(isnan(param.NaNColor)) ... NaNColor is specified
                         || ~param.IsGraphics2 && ~param.UseFigureColormap;


% if param.IsGraphics2 && ~param.UseFigureColormap && ~isempty(param.ColorBar) % graphics v2
%     warning('heatmap:graphics2figurecolormap', 'The UseFigureColormap false option with colorbar is not supported in versions R2014b and above. In most such cases UseFigureColormap false is unnecessary');
% end
    

end

% Visualize heatmap image
function p = plotHeatmap(p, mat)

p.cdata = [];
if p.UseLogColormap
    p.Colormap = resamplecmap(p.Colormap, p.ColorLevels, ...
                           logspace(0,log10(p.ColorLevels),p.ColorLevels));
end

if p.ExplicitlyComputeImage
    % Calculate the color data explicitly and then display it as an image.
    n = p.MinColorValue;
    x = p.MaxColorValue;
    if x == n, x = n+1; end
    p.cdata = round((mat-n)/(x-n)*(p.ColorLevels-1)+1);
    %p.cdata = ceil((mat-n)/(x-n)*p.ColorLevels);
    p.cdata(p.cdata<1) = 1; % Clipping
    p.cdata(p.cdata>p.ColorLevels) = p.ColorLevels; % Clipping
    nanInd = find(isnan(p.cdata));
    p.cdata(isnan(p.cdata)) = 1;
    p.cdata = reshape(p.Colormap(p.cdata(:),:),[size(p.cdata) 3]);
    % Handle NaNColor case
    if ~all(isnan(p.NaNColor))
        p.cdata(nanInd                     ) = p.NaNColor(1); % Set red   color level of nan indices
        p.cdata(nanInd +   numel(p.cdata)/3) = p.NaNColor(2); % Set green color level of nan indices
        p.cdata(nanInd + 2*numel(p.cdata)/3) = p.NaNColor(3); % set blue  color level of nan indices
    end
    % Add a small dummy image so that colorbar subsequently works
    [indr, indc] = find(~isnan(mat),1);
    imagesc(indr, indc, mat(indr,indc),'Parent',p.hAxes);
    nextplot = get(p.hAxes,'nextplot');
    set(p.hAxes,'nextplot','add');
    p.hImage = image(p.cdata, 'Parent', p.hAxes);
    set(p.hAxes,'nextplot',nextplot);
    axis(p.hAxes,'tight');
else
    % Use a scaled image plot. Axes CLims and colormap will be set later
    p.hImage = imagesc(mat, 'Parent', p.hAxes);
end

set(p.hAxes, 'CLim', [p.MinColorValue p.MaxColorValue]); % Ensure proper clipping for colorbar
if p.UseFigureColormap
    set(p.hFig,'Colormap',p.Colormap);
elseif p.IsGraphics2
    % Set the axes colormap and limits
    colormap(p.hAxes, p.Colormap);
    %set(p.hAxes, 'CLim', [p.MinColorValue p.MaxColorValue]);
end
end

% Generate grid lines
function generateGridLines(p)
if ~strcmp(p.GridLines,'none')
    xlim = get(p.hAxes,'XLim');
    ylim = get(p.hAxes,'YLim');
    for i = 1:diff(xlim)-1
        line('Parent',p.hAxes,'XData',[i i]+.5, 'YData', ylim, 'LineStyle', p.GridLines);
    end
    for i = 1:diff(ylim)-1
        line('Parent',p.hAxes,'XData',xlim, 'YData', [i i]+.5, 'LineStyle', p.GridLines);
    end
end
end

% Add color bar
function addColorbar(p, mat, textmat)

if isempty(p.Colorbar)
    return;
elseif iscell(p.Colorbar)
    c = colorbar(p.Colorbar{:});
else
    c = colorbar;
end
if p.IsGraphics2
    c.Limits = p.hAxes.CLim;
    ticks = get(c,'Ticks');
else
    if p.ExplicitlyComputeImage || ~p.UseFigureColormap
        d = findobj(get(c,'Children'),'Tag','TMW_COLORBAR'); % Image
        set(d,'YData', get(p.hAxes,'CLim'));
        set(c,'YLim', get(p.hAxes,'CLim'));
    end
    ticks = get(c,'YTick');
    tickAxis = 'Y';
    if isempty(ticks)
        ticks = get(c,'XTick');
        tickAxis = 'X';
    end
end
  
if ~isempty(ticks)
    
    if ischar(textmat) % If format string, format colorbar ticks in the same way
        ticklabels = arrayfun(@(x){sprintf(textmat,x)},ticks);
    else
        ticklabels = num2str(ticks(:));
    end
    if p.IsGraphics2
        set(c, 'TickLabels', ticklabels);
    else
        set(c, [tickAxis 'TickLabel'], ticklabels);
    end
    
end


end


% ------------------------- Tick Label Functions -------------------------

% Set axes tick labels
function [p, xlab, ylab, hXText, origPos] = setAxesTickLabels(p, xlab, ylab)

if isempty(p.axesInfo) % Not previously a heatmap axes
    origPos = [get(p.hAxes,'Position') get(p.hAxes,'OuterPosition')];
else
    origPos = p.axesInfo.origAxesPos;
    set(p.hAxes, 'Position', origPos(1:4), 'OuterPosition', origPos(5:8));
end


if isempty(p.TickFontSize)
    p.TickFontSize = get(p.hAxes, 'FontSize');
else
    set(p.hAxes, 'FontSize', p.TickFontSize);
end

if isempty(ylab) % No ticks or labels
    set(p.hAxes,'YTick',[],'YTickLabel','');
else
    if isnumeric(ylab) % Numeric tick labels
        ylab = arrayfun(@(x){num2str(x)},ylab);
    end
    if ischar(ylab)
        ylab = cellstr(ylab);
    end
    ytick = get(p.hAxes, 'YTick'); 
    ytick(ytick<1|ytick>length(ylab)) = [];
    if p.ShowAllTicks || length(ytick) > length(ylab)
        ytick = 1:length(ylab);
    end
    set(p.hAxes,'YTick',ytick,'YTickLabel',ylab(ytick));
end
if p.IsGraphics2
    if p.TickTexInterpreter
        set(p.hAxes,'TickLabelInterpreter','tex');
    else
        set(p.hAxes,'TickLabelInterpreter','none');
    end
end
% Xlabels are trickier because they could have a TickAngle
hXText = []; % Default value
if isempty(xlab)
    set(p.hAxes,'XTick',[],'XTickLabel','');
else
    if isnumeric(xlab)
        xlab = arrayfun(@(x){num2str(x)},xlab);
    end
    if ischar(xlab)
        xlab = cellstr(xlab);
    end
    xtick = get(p.hAxes, 'XTick');
    xtick(xtick<1|xtick>length(xlab)) = [];
    if p.ShowAllTicks || length(xtick) > length(xlab)
        xtick = 1:length(xlab);
    end
    if p.IsGraphics2
        set(p.hAxes,'XTick',xtick,'XTickLabel',xlab(xtick),'XTickLabelRotation', p.TickAngle);
    else
        if p.TickAngle == 0
            set(p.hAxes,'XTick',xtick,'XTickLabel',xlab(xtick));
        else
            hXText = createXTicks(p.hAxes, p.TickAngle, xtick, xlab(xtick), p.TickTexInterpreter);
            adjustAxesToAccommodateTickLabels(p.hAxes, hXText);
        end
    end
end



end

% Create Rotated X Tick Labels (Graphics v1)
function hXText = createXTicks(hAxes, tickAngle, xticks, xticklabels, texInterpreter)

    axXLim = get(hAxes, 'XLim');
    [xPos, yPos] = calculateTextTickPositions(hAxes, axXLim, xticks);
    
    if texInterpreter
        interpreter = 'tex';
    else
        interpreter = 'none';
    end
    hXText = text(xPos, yPos, cellstr(xticklabels), 'Units', 'normalized', ...
                  'Parent', hAxes, 'FontSize', get(hAxes,'FontSize'), ...
                  'HorizontalAlignment', 'right', 'Rotation', tickAngle,...
                  'Interpreter', interpreter);
                
    set(hAxes, 'XTick', xticks, 'XTickLabel', '');

end

% Calculate positions of X tick text objects in normalized units
function [xPos, yPos] = calculateTextTickPositions(hAxes, xlim, ticks)
oldunits = get(hAxes,'Units');
set(hAxes,'units','pixels');
axPos = get(hAxes,'position');
set(hAxes,'units',oldunits);

xPos = (ticks - xlim(1))/diff(xlim);
%yPos = -.08 * ones(size(xPos));
yPos = -7.82/axPos(4) * ones(size(xPos));
end

% Adjust axes and tick positions so that everything fits well on screen
function adjustAxesToAccommodateTickLabels(hAxes, hXText)
% The challenge here is that the axes container, especially in a subplot is
% not well defined. The outer position property does not fully span or
% contain the x tick text objects. So here we just shrink the axes height
% just a little so that the axes and tick labels take the same room as the
% axes would have without the ticks.

[axPosP, axPosN, axOPP, axOPN, coPosP, textPosP]  = ...
    getGraphicsObjectsPositions(hAxes, hXText); %#ok<ASGLU>

header = axOPP(4) + axOPP(2) - axPosP(4) - axPosP(2); % Distance between top of axes and container in pixels; 
delta = 5; % To adjust for overlap between area designated for regular ticks and area occupied by rotated ticks
axHeightP = axOPP(4) - header - delta - textPosP(4);

% Fudge axis position if labels are taking up too much room
if textPosP(4)/(textPosP(4)+axHeightP) > .7 % It's taking up more than 70% of total height
    axHeightP = (1/.7-1) * textPosP(4); % Minimum axis 
end
axHeightN = max(0.0001, axHeightP / coPosP(4));

axPosN = axPosN + [0 axPosN(4)-axHeightN 0 axHeightN-axPosN(4)];
set(hAxes,'Position', axPosN)

end

% Calculate graphics objects positions in pixels and normalized units
function [axPosP, axPosN, axOPP, axOPN, coPosP, textPosP, textPosN] =...
         getGraphicsObjectsPositions(hAxes, hXText)

axPosN = get(hAxes, 'Position'); axOPN = get(hAxes, 'OuterPosition');
set(hAxes,'Units','Pixels');
axPosP = get(hAxes, 'Position'); axOPP = get(hAxes, 'OuterPosition');
set(hAxes,'Units','Normalized');

hContainer = get(hAxes,'Parent');
units = get(hContainer,'Units');
set(hContainer,'Units','Pixels');
coPosP = get(hContainer,'Position');
set(hContainer,'Units',units);

set(hXText,'Units','pixels'); % Measure height in pixels
extents = get(hXText,'Extent'); % Get heights for all text objects
extents = vertcat(extents{:}); % Collect heights in one matrix
textPosP = [min(extents(:,1)) min(extents(:,2)) ...
    max(extents(:,3)+extents(:,1))-min(extents(:,1)) ...
    max(extents(:,4))]; % Find dimensions of text label block
set(hXText,'Units','normalized'); % Restore previous behavior
extents = get(hXText,'Extent'); % Get heights for all text objects
extents = vertcat(extents{:}); % Collect heights in one matrix
textPosN = [min(extents(:,1)) min(extents(:,2)) ...
    max(extents(:,3)+extents(:,1))-min(extents(:,1)) ...
    max(extents(:,4))]; % Find dimensions of text label block

end


% -------------------------- Callback Functions --------------------------

% Update x-, y- and text-labels with respect to axes limits
function updateLabels(hAxes, axesLimitsChanged)

axInfo = getHeatmapAxesInfo(hAxes);
if isempty(axInfo), return; end
p = axInfo.Parameters;

% Update text font size to fill the square
if ~isempty(p.hText) && ishandle(p.hText(1))
    fs = axInfo.FontScaleFactor * getBestFontSize(hAxes);
    if fs > 0
        set(p.hText,'fontsize',fs,'visible','on');
    else
        set(p.hText,'visible','off');
    end
end

if axesLimitsChanged && ~isempty(axInfo.displayText) % If limits change & text labels are displayed
    % Get positions of text objects
    textPos = get(p.hText,'Position');
    textPos = vertcat(textPos{:});
    % Get axes limits
    axXLim = get(hAxes, 'XLim');
    axYLim = get(hAxes, 'YLim');
    % Find text objects within axes limit
    ind = textPos(:,1) > axXLim(1) & textPos(:,1) < axXLim(2) & ...
        textPos(:,2) > axYLim(1) & textPos(:,2) < axYLim(2);
    set(p.hText(ind), 'Visible', 'on');
    set(p.hText(~ind), 'Visible', 'off');
end
    
% Modify Y Tick Labels

if ~isempty(axInfo.ylab)
    axYLim = get(hAxes, 'YLim');
    if p.ShowAllTicks
        yticks = ceil(axYLim(1)):floor(axYLim(2));
    else
        set(hAxes, 'YTickMode', 'auto');
        yticks = get(hAxes, 'YTick');
        yticks = yticks( yticks == floor(yticks) );
        yticks(yticks<1|yticks>length(axInfo.ylab)) = [];
    end
    ylabels = repmat({''},1,max(yticks));
    ylabels(1:length(axInfo.ylab)) = axInfo.ylab;
    set(hAxes, 'YTick', yticks, 'YTickLabel', ylabels(yticks));
end
    
if ~isempty(axInfo.xlab)    
    
    axXLim = get(hAxes, 'XLim');

    if p.ShowAllTicks
        xticks = ceil(axXLim(1)):floor(axXLim(2));
    else
        set(hAxes, 'XTickMode', 'auto');
        xticks = get(hAxes, 'XTick');
        xticks = xticks( xticks == floor(xticks) );
        xticks(xticks<1|xticks>length(axInfo.xlab)) = [];
    end
    xlabels = repmat({''},1,max(xticks));
    xlabels(1:length(axInfo.xlab)) = axInfo.xlab;
    
    if ~isempty(axInfo.hXText) % Rotated X tick labels exist
        try delete(axInfo.hXText); end %#ok<TRYNC>
        axInfo.hXText = createXTicks(hAxes, p.TickAngle, xticks, xlabels(xticks), p.TickTexInterpreter);
        set(hAxes, 'UserData', axInfo);
    else
        set(hAxes, 'XTick', xticks, 'XTickLabel', xlabels(xticks));
    end
    
    %adjustAxesToAccommodateTickLabels(hAxes, axInfo.hXText)
end

end

% Callback for data cursor
function output_txt = cursorFun(obj, eventdata)
hAxes = ancestor(eventdata.Target, 'axes');
axInfo = getHeatmapAxesInfo(hAxes);

pos = eventdata.Position;

if ~isempty(axInfo)
    
    try
        val = axInfo.displayText{pos(2), pos(1)};
    catch %#ok<CTCH>
        val = num2str(axInfo.mat(pos(2), pos(1)));
    end
    if isempty(axInfo.xlab), i = int2str(pos(1)); else, i = axInfo.xlab{pos(1)}; end
    if isempty(axInfo.ylab), j = int2str(pos(2)); else, j = axInfo.ylab{pos(2)}; end
    output_txt = sprintf('X: %s\nY: %s\nVal: %s', i, j, val);
    
else
    if length(pos) == 2
        output_txt = sprintf('X: %0.4g\nY: %0.4g', pos(1), pos(2));
    else
        output_txt = sprintf('X: %0.4g\nY: %0.4g\nZ: %0.4g', pos(1), pos(2), pos(3));
    end
end
end

% Callback for resize event
function resize(obj, evd)
hAxes = findobj(obj, 'type', 'axes');
for i = 1:length(hAxes)
    updateLabels(hAxes(i), false); 
end
end

% Extract heatmap parameters for callback
function axInfo = getHeatmapAxesInfo(axH)
axInfo = get(axH, 'UserData');
try
    if ~strcmp(axInfo.Type, 'heatmap')
        axInfo = [];
    end
catch %#ok<CTCH>
    axInfo = [];
end
end


% ------------------------- Text Label Functions -------------------------

% Create text labels
function [p, displaytext, factor] = setTextLabels(p, mat, textmat)

if isempty(textmat)
    p.hText = [];
    displaytext = {};
    factor = 0;
    return
end
    
if isscalar(textmat) && textmat % If true convert mat to text
    displaytext = arrayfun(@(x){num2str(x)},mat);
elseif ischar(textmat) % If a format string, convert mat to text with specific format
    displaytext = arrayfun(@(x){sprintf(textmat,x)},mat);
elseif isnumeric(textmat) && numel(textmat)==numel(mat) % If numeric, convert to text
    displaytext = arrayfun(@(x){num2str(x)},textmat);
elseif iscellstr(textmat) && numel(textmat)==numel(mat) % If cell array of strings, it is already formatted
    displaytext = textmat;
else
    error('texmat is incorrectly specified');
end

if ischar(p.TextColor) && strcmp(p.TextColor,'xor')
    colorprop = 'EraseMode';
else
    colorprop = 'Color';
end

autoFontSize = getBestFontSize(p.hAxes);
if isempty(p.FontSize)
    p.FontSize = autoFontSize;
end
[xpos,ypos] = meshgrid(1:size(mat,2),1:size(mat,1));
if p.FontSize > 0
    p.hText = text(xpos(:),ypos(:),displaytext(:),'FontSize',p.FontSize,...
        'HorizontalAlignment','center', colorprop, p.TextColor,'Parent',p.hAxes);
else
    p.hText = text(xpos(:),ypos(:),displaytext(:),'Visible','off',...
        'HorizontalAlignment','center', colorprop, p.TextColor,'Parent',p.hAxes);
end

% Calculate factor to scale font size in future callbacks
factor = p.FontSize/autoFontSize;
if isnan(factor), factor = 1; end
if isinf(factor), factor = p.FontSize/6; end

% % Set up listeners to handle appropriate zooming
% addlistener(p.hAxes,{'XLim','YLim'},'PostSet',@(obj,evdata)resizeText);
% try
%     addlistener(p.hFig,'SizeChange',@(obj,evdata)resizeText);
% catch
%     addlistener(p.hFig,'Resize',@(obj,evdata)resizeText);
% end



%     function resizeText
%         if ~isempty(hText) && ishandle(hText(1))
%             fs = factor*getBestFontSize(hAxes);
%             if fs > 0
%                 set(hText,'fontsize',fs,'visible','on');
%             else
%                 set(hText,'visible','off');
%             end
%         end
%     end

end

% Guess best font size from axes size using heuristics
function fs = getBestFontSize(imAxes)

hFig = ancestor(imAxes,'figure');
magicNumber = 80;
nrows = diff(get(imAxes,'YLim'));
ncols = diff(get(imAxes,'XLim'));
if ncols < magicNumber && nrows < magicNumber
    ratio = max(get(hFig,'Position').*[0 0 0 1])/max(nrows,ncols);
elseif ncols < magicNumber
    ratio = max(get(hFig,'Position').*[0 0 0 1])/ncols;
elseif nrows < magicNumber
    ratio = max(get(hFig,'Position').*[0 0 0 1])/nrows;
else
    ratio = 1;
end
fs = min(9,ceil(ratio/4));    % the gold formula
if fs < 4
    fs = 0;
end
end

% -------------------------- Colormap Functions --------------------------

% Determine the colormap to use
function p = calculateColormap(p, mat)

if isempty(p.Colormap)
    if p.IsGraphics2 && ~p.UseFigureColormap
        p.Colormap = colormap(p.hAxes);
    else
        p.Colormap = get(p.hFig,'Colormap');
    end
    if isempty(p.ColorLevels)
        p.ColorLevels = size(p.Colormap,1);
    else
        p.Colormap = resamplecmap(p.Colormap, p.ColorLevels);
    end
elseif ischar(p.Colormap) || isa(p.Colormap,'function_handle')
    if isempty(p.ColorLevels), p.ColorLevels = 64; end
    if strcmp(p.Colormap, 'money')
        p.Colormap = money(mat, p.ColorLevels);
    else
        p.Colormap = feval(p.Colormap,p.ColorLevels);
    end
elseif iscell(p.Colormap)
    p.Colormap = feval(p.Colormap{1}, p.Colormap{2:end});
    p.ColorLevels = size(p.Colormap,1);
elseif isnumeric(p.Colormap) && size(p.Colormap,2) == 3
    p.ColorLevels = size(p.Colormap,1);
else
    error('Incorrect value for colormap parameter');
end % p.Colormap is now a p.ColorLevels-by-3 rgb vector
assert(p.ColorLevels == size(p.Colormap,1));

end

% Resample a colormap by interpolation or decimation
function cmap = resamplecmap(cmap, clevels, xi)

t = cmap;
if nargin < 3
    xi = linspace(1,clevels,size(t,1)); 
end
xi([1 end]) = [1 clevels]; % These need to be exact for the interpolation to 
% work and we don't want machine precision messing it up
cmap = [interp1(xi, t(:,1), 1:clevels);...
        interp1(xi, t(:,2), 1:clevels);...
        interp1(xi, t(:,3), 1:clevels)]';
end

% Generate Red-White-Green color map
function cmap = money(data, clevels)
% Function to make the heatmap have the green, white and red effect
n = min(data(:));
x = max(data(:));
if x == n, x = n+1; end
zeroInd = round(-n/(x-n)*(clevels-1)+1);
if zeroInd <= 1 % Just green
    b = interp1([1 clevels], [1 0], 1:clevels);
    g = interp1([1 clevels], [1 1], 1:clevels);
    r = interp1([1 clevels], [1 0], 1:clevels);
elseif zeroInd >= clevels % Just red
    b = interp1([1 clevels], [0 1], 1:clevels);
    g = interp1([1 clevels], [0 1], 1:clevels);
    r = interp1([1 clevels], [1 1], 1:clevels);
else
    b = interp1([1 zeroInd clevels], [0 1 0], 1:clevels); 
    g = interp1([1 zeroInd clevels], [0 1 1], 1:clevels);
    r = interp1([1 zeroInd clevels], [1 1 0], 1:clevels);
end

cmap = [r' g' b'];
end

% Generate Red-White color map
function cmap = red(levels)
r = ones(levels, 1);
g = linspace(1, 0, levels)'; 
cmap = [r g g];
end






%#ok<*INUSD>
%#ok<*DEFNU>
%#ok<*INUSL>
