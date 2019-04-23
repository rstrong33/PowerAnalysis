function power_results = PowerAnalysis_ANOVA(prefs)

prefs.max_array_size = 1000000;
prefs.figure_width = 700;
prefs.figure_height = 700;

%load csv
csvdata = textscan(fopen(prefs.csv_file), '%s%s%s%s', 'delimiter', ',');

%organize data
header = {csvdata{1}{1}, csvdata{2}{1}, csvdata{3}{1}, csvdata{4}{1}};
sub_list = csvdata{1}(2:end);
ans_list = csvdata{2}(2:end);
f1_levels_list = csvdata{3}(2:end);
f2_levels_list = csvdata{4}(2:end);
ans_as_nums = str2num(cell2mat(ans_list));

%convert subject names to integers
sub_names = unique(csvdata{1}(2:end), 'stable');
sub_nums = zeros(length(csvdata{1}(2:end)), 1);
for sn = 1:length(sub_names)
    sub_nums(strcmp(sub_list, sub_names(sn)), 1) = sn;
end

%convert factor levels to integers
%f1
f1_names = unique(csvdata{3}(2:end));
f1_levels = zeros(length(csvdata{3}(2:end)), 1);
f1_num_levels = length(f1_names);
for cn = 1:f1_num_levels
    f1_levels(strcmp(f1_levels_list, f1_names(cn)), 1) = cn;
end

%f2
f2_names = unique(csvdata{4}(2:end));
f2_levels = zeros(length(csvdata{4}(2:end)), 1);
f2_num_levels = length(f2_names);
for cn = 1:f2_num_levels
    f2_levels(strcmp(f2_levels_list, f2_names(cn)), 1) = cn;
end

%subs, scores, and conditions as numbers
prefs.data = [sub_nums, ans_as_nums, f1_levels, f2_levels];

%column headers from CSV data
prefs.header = header;

%what kind of design (between, within, mixed)
if length(unique(f1_levels(sub_nums==1))) == 1 && length(unique(f2_levels(sub_nums==1))) == 1
    prefs.between_within_mixed = 1;
elseif length(unique(f1_levels(sub_nums==1))) == 1 || length(unique(f2_levels(sub_nums==1))) == 1
    
    prefs.between_within_mixed = 3;
    %make factor 1 the between subjects factor (if it isn't already)
    if length(unique(f1_levels(sub_nums==1))) > 1
        prefs.data = [sub_nums, ans_as_nums, f2_levels, f1_levels];
        tmp_names = f1_names;
        tmp_num_levels = f1_num_levels;
        f1_names = f2_names;
        f1_num_levels = f2_num_levels;
        f2_names = tmp_names;
        f2_num_levels = tmp_num_levels;
        prefs.header(3:4) = prefs.header([4,3]);
    end
elseif length(unique(f2_levels(sub_nums==1))) > 1 && length(unique(f2_levels(sub_nums==1))) > 1
    prefs.between_within_mixed = 2;
else
    error('Error in Data File')
end

%condition names (original as text)
prefs.f1_names = f1_names;
prefs.f2_names = f2_names;
prefs.f1_num_levels = f1_num_levels;
prefs.f2_num_levels = f2_num_levels;
prefs.sub_nums = sub_nums;

%determine whether DV is percent correct (only 2 choices)
if length(unique(prefs.data(:,2))) == 2
    %see if we can do fast sim (depends on subjects/trials/conditions)
    tMax = max(prefs.trial_range);
    nMax = max(prefs.N_range);
    nConds = prefs.f1_num_levels * prefs.f2_num_levels;
    
   
    if prefs.between_within_mixed == 2  %within-subjects design
        try
            zeros(nMax, (tMax+1)^nConds);
            fast_sim = true;
        catch
            fast_sim = false;
        end
    elseif prefs.between_within_mixed == 3  %mixed-factor design
        try
            zeros(nMax, (tMax+1)^prefs.f2_num_levels);
            fast_sim = true;
        catch
            fast_sim = false;
        end
    end
else
    fast_sim = false;
end


disp('Loading Data...');

%make plot of pilot data and simulation parameters
prefs = DoPilotPlot(prefs);

%main power analysis
if prefs.between_within_mixed == 2
    if fast_sim
        power_results = PowerAnalysisWithinFast(prefs);
    else
        power_results = PowerAnalysisWithin(prefs);
    end
elseif prefs.between_within_mixed == 1
    power_results = PowerAnalysisBetween(prefs);
elseif prefs.between_within_mixed == 3
    if fast_sim
        power_results = PowerAnalysisMixedFast(prefs);
    else
        power_results = PowerAnalysisMixed(prefs);
    end
end

end

function prefs = DoPilotPlot(prefs)


%get condition means
nConds = prefs.f1_num_levels * prefs.f2_num_levels;
sub_means = nan(length(unique(prefs.sub_nums)), nConds);


for s = 1:length(unique(prefs.sub_nums))
    cond = 0;
    for f1 = 1:prefs.f1_num_levels
        for f2 = 1:prefs.f2_num_levels
            cond = cond + 1;
            sub_means(s, cond) = mean(prefs.data(prefs.data(:,1) == s & ...
                prefs.data(:,3) == f1 & prefs.data(:,4) == f2, 2));
        end
    end
end

prefs.sub_means = sub_means;

%condition number labels
color_matrix = [.75 .75 .75;
    1 1 1;
    1 0 0;
    0 1 0;
    0 0 1;
    0 0 0;
    0 1 1];

count = 0;
xspot = 1;
graphx = [];
graph_colors = [];
mid_spot = [];
condition_means = nanmean(sub_means);

for f1 = 1:length(prefs.f1_names)
    for f2 = 1:length(prefs.f2_names)
        count = count + 1;
        level_f1(count) = f1;
        level_f2(count) = f2;
        graph_colors = [graph_colors; color_matrix(f2, :)];
    end
    mid_spot = [mid_spot, mean(xspot:xspot+length(prefs.f2_names)-1)];
    graphx = [graphx, xspot:xspot+length(prefs.f2_names)-1];
    xspot = xspot + length(prefs.f2_names) + 1;
end

figure(1)

clf
hold on
subplot(2,1,1)
hold on
title('Pilot Data')
for c = 1:count
    bar(graphx(c), condition_means(c), 'FaceColor', graph_colors(c,:))
end

range = max(condition_means) - min(condition_means);
if range == 0
    range = max(condition_means);
end
drop = .5;
ymin = min(condition_means) - drop*range;
ymax = max(condition_means) + drop*range;
ylim([ymin, ymax])
label_y = min(condition_means) - (drop/2)*range;
xlim([min(graphx) - 1, max(graphx) + 1])

if rem(length(prefs.f2_names), 2) == 1 %odd amount of f2 levels
    range = min(graphx) - 1: max(graphx) + 1;
    for c = 1:length(range)
        x_label{c} = '';
    end
    for c = 1:length(prefs.f1_names)
        x_label{range == mid_spot(c)} = prefs.f1_names{c};
    end
    set(gca,'XTick',range)
    set(gca, 'XTickLabel',x_label, 'fontsize',24)
else
    range = min(graphx) - 1:.5: max(graphx) + 1;
    for c = 1:length(range)
        x_label{c} = '';
    end
    for c = 1:length(prefs.f1_names)
        x_label{range == mid_spot(c)} = prefs.f1_names{c};
    end
    set(gca,'XTick',range)
    set(gca, 'XTickLabel',x_label, 'fontsize',24)
end

legend(prefs.f2_names, 'FontSize', 16)

ylabel(prefs.header{2})

%bar labels
fs = 16;
for b = 1:length(condition_means)
    text(graphx(b), label_y, num2str(b), 'FontSize', fs);
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
%ME 1
if prefs.sig_ME1
    count = count + 1;
    spot = spot - spot_jump;
    
    txt = [num2str(count), ': Main Effect ', prefs.header{3}];
    
    text(.2, spot, txt, 'FontSize', 16);
end

%ME2
if prefs.sig_ME2
    count = count + 1;
    spot = spot - spot_jump;
    
    txt = [num2str(count), ': Main Effect ', prefs.header{4}];
    text(.2, spot, txt, 'FontSize', 16);
end

%interaction
if prefs.sig_int
    count = count + 1;
    spot = spot - spot_jump;
    
    txt = [num2str(count), ': Interaction of ',  prefs.header{3}, ' x ', prefs.header{4}];
    
    text(.2, spot, txt, 'FontSize', 16);
end

%t-tests
for p = 1:size(prefs.comps, 1)
    count = count + 1;
    spot = spot - spot_jump;
    
    %extra text
    if prefs.between_within_mixed == 3
        if level_f1(prefs.comps(p, 1)) == level_f1(prefs.comps(p, 2))
            extra_text = ' (within subjects)';
        else
            extra_text = ' (between subjects)';
        end
    elseif prefs.between_within_mixed == 2
        extra_text = ' (within subjects)';
    end
    
    txt = [num2str(count), ': ', num2str(prefs.comps(p, 1)), ' > ', num2str(prefs.comps(p, 2)), extra_text];
    text(.2, spot, txt, 'FontSize', 16);
end

%position figure on screen
set(0,'units','pixels')
Pix_SS = get(0,'screensize');
w = prefs.figure_width;
h = prefs.figure_height;
set(gcf, 'Position',  [Pix_SS(3)-1.75*w, Pix_SS(4) - h, w, h])


end

%within subjects design function
function power_results = PowerAnalysisWithinFast(prefs)

%simulation info
nSims = prefs.nSims; %number of experiments to simulate
nPilotSubs = length(unique(prefs.data(:,1))); %how many subjects in actual data
sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest
nConds = prefs.f1_num_levels*prefs.f2_num_levels; %number of conditions

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
            cond_probs{c} = binopdf(0:t, t, prefs.sub_means(n,c));
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
        pc = round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)));
        disp([num2str(pc), '% Complete']);
        
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
        if prefs.sig_ME1 || prefs.sig_ME2 || prefs.sig_int
            %anova part
            Y = [];
            for c = 1:nConds
                Y = [Y;reshape(cond_score{c}(outcome_samples), nSubs, nSims)];
            end
            S = repmat(1:nSubs, 1, nConds)';
            
            F1 = [ones(nSubs * nConds/2, 1); 1 + ones(nSubs * nConds/2, 1)];
            F2 = repmat([ones(nSubs, nConds/4); 1 + ones(nSubs, nConds/4)], 2, 1);
            
            stats = rm_anova2_matrix(Y,S,F1,F2);
            
            if prefs.sig_ME1
                power_marker(end+1,:) = stats.pA < prefs.alpha;
            end
            if prefs.sig_ME2
                power_marker(end+1,:) = stats.pB < prefs.alpha;
            end
            if prefs.sig_int
                power_marker(end+1,:) = stats.pAB < prefs.alpha;
            end
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

%make heat map plot
MakeHeatMap(power_results)

end

function power_results = PowerAnalysisWithin(prefs)

%simulation info
nSims = prefs.nSims; %number of experiments to simulate
nPilotSubs = length(unique(prefs.data(:,1))); %how many subjects in actual data
sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest
nConds = prefs.f1_num_levels*prefs.f2_num_levels; %number of conditions
nPilotTrials = sum(prefs.data(:,1) == 1 & prefs.data(:,3) == 1 & prefs.data(:,4) == 1);

%organize data sub*trial*cond
pilot_data = nan(nPilotSubs, nPilotTrials, nConds);
for s = 1:nPilotSubs
    cond = 0;
    for f1 = 1:prefs.f1_num_levels
        for f2 = 1:prefs.f2_num_levels
            cond = cond + 1;
            pilot_data(s,:,cond) = prefs.data(prefs.data(:,1) == s & prefs.data(:,3) == f1 & prefs.data(:,4) == f2, 2)';
        end
    end
end

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));
dz_vect = cell(1, nComps);

for trial_count= 1:length(trial_vector)
    
    t = trial_vector(trial_count);
    
    %calculate measurement variability for each subject based upon
    %condition means and number of trials
    
    %figure out sampling variability based upon number of subjects
    for sub_count = 1:length(sub_vector)
        
        clc
        pc = round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)));
        disp([num2str(pc), '% Complete']);
        
        
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
        
        if prefs.sig_ME1 || prefs.sig_ME2 || prefs.sig_int
            %anova part
            
            Y = [];
            F1 = [];
            F2 = [];
            c = 0;
            for f1 = 1:prefs.f1_num_levels
                F1 = [F1; f1*ones(nSubs * prefs.f2_num_levels,1)];
                for f2 = 1:prefs.f2_num_levels
                    c = c + 1;
                    Y = [Y; cond_scores{c}];
                    F2 = [F2; f2*ones(nSubs,1)];
                end
            end
            
            S = repmat(1:nSubs, 1, nConds)';
            
            stats = rm_anova2_matrix(Y,S,F1,F2);
            
            if prefs.sig_ME1
                power_marker(end+1,:) = stats.pA < prefs.alpha;
            end
            if prefs.sig_ME2
                power_marker(end+1,:) = stats.pB < prefs.alpha;
            end
            if prefs.sig_int
                power_marker(end+1,:) = stats.pAB < prefs.alpha;
            end
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

%make heat map plot
MakeHeatMap(power_results)

end

function power_results = PowerAnalysisBetween(prefs)
%check to see if data is in decimal form (not percentage)
%if so, divide by 100
if sum(prefs.condition_allocation) ~= 1
    error('Condition allocation does not add up to 1 (100%). Use fractions if this is due to rounding error (e.g., use 1/3 instead of .33)')
elseif length(prefs.condition_allocation) ~= prefs.f1_num_levels
    error('Must have an allocation amount for each between-subjects level.')
end

%simulation info
nSims = prefs.nSims; %number of experiments to simulate
sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest
nConds = size(prefs.data,2); %number of conditions

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
subs_by_cond = cell(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));
ds_vect = cell(1, nComps);
nPilotSubs = zeros(1, nConds);
for c = 1:nConds
    nPilotSubs(c) = sum(~isnan(prefs.data(:,c))); %how many subjects in actual data per condition
end


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
            cond_prob{c}(n,:) = binopdf(0:t, t, prefs.data(n,c));
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

%plot
figure(1)
clf

power = round(power, 2);
if all(all(power == 0))
    xlim([0,1])
    ylim([0,1])
    text(.27, .5,'Cannot Display Heatmap', 'FontSize', 20)
    text(.2, .4,'Power for all study designs is 0%', 'FontSize', 20)
else
    
    heatmap(power, sub_vector, trial_vector, true, 'GridLines', '-', 'FontSize', 14);
    xlabel('Total # of Subjects', 'FontSize', 20)
    ylabel('# of Trials Per Condition', 'FontSize', 20)
    title('Power by N and # of Trials', 'FontSize', 20)
end
end

function power_results = PowerAnalysisMixedFast(prefs)

if sum(prefs.condition_allocation) ~= 1
    error('Condition allocation does not add up to 1 (100%). Use fractions if this is due to rounding error (e.g., use 1/3 instead of .33)')
elseif length(prefs.condition_allocation) ~= prefs.f1_num_levels
    error('Must have an allocation amount for each between-subjects level.')
end

%simulation info
nSims = prefs.nSims; %number of experiments to simulate

%number of subs for each between-subs factor level

num_between_levels = prefs.f1_num_levels;
num_within_levels =  prefs.f2_num_levels;

cond = 0;
for f1 = 1:num_between_levels
    for f2 = 1:num_within_levels
        cond = cond + 1;
        level_between(cond) = f1;
        level_within(cond) = f2;
    end
end


for c = 1:num_between_levels
    between_subs{c} = unique(prefs.sub_nums(prefs.data(:,3) == c))';
    nPilotSubs(c) = length(between_subs{c});
end

sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));

%get subject-level long-form data for simulation
prefs.subs = [];
prefs.F1 = [];
prefs.F2 = [];
prefs.Y = [];

for f1 = 1:num_between_levels
    for sub = 1:nPilotSubs(f1)
        for f2 = 1:num_between_levels
            prefs.subs = [prefs.subs; between_subs{f1}(sub)];
            prefs.F1 = [prefs.F1; f1];
            prefs.F2 = [prefs.F2; f2];
            prefs.Y = [prefs.Y; mean(prefs.data(prefs.data(:,1) == between_subs{f1}(sub) & prefs.data(:,4) == f2,2))];
        end
    end
end


for trial_count= 1:length(trial_vector)
    
    t = trial_vector(trial_count);
    
    
    for between_level = 1:num_between_levels
        %determine condition difference pdf for each subject in pilot data
        cFinal{between_level} = zeros(nPilotSubs(between_level), (t+1)^num_within_levels);
        outcomes = 1:(t+1)^num_within_levels;
        
        %calculate measurement variability for each subject based upon
        %condition means and number of trials
        for n = 1:nPilotSubs(between_level)
            
            cond_probs = cell(1, num_within_levels);
            
            for c = 1:num_within_levels
                cond_probs{between_level, c} = binopdf(0:t, t, prefs.Y(prefs.subs == between_subs{between_level}(n) & prefs.F1 == between_level & prefs.F2 == c));
            end
            
            %first two conditions
            tmp =  cond_probs{between_level, 1}' * cond_probs{between_level, 2};
            tmp = reshape(tmp, numel(tmp), 1);
            
            %additional conditions
            if num_within_levels > 2
                for c = 3:num_within_levels
                    tmp = reshape(tmp*cond_probs{between_level, c}, numel(tmp)*length(cond_probs{between_level, c}), 1);
                end
            end
            
            %cond_score = cell(1, nConds);
            for c = 1:num_within_levels
                cond_score{between_level, c} = repmat(repelem(0:t, (t+1)^(c-1))', (t+1)^(num_within_levels-c), 1);
            end
            
            cFinal{between_level}(n, :) = tmp;
            
        end
    end
    
    
    %figure out sampling variability based upon number of subjects
    for sub_count = 1:length(sub_vector)
        
        clc
        pc = round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)));
        disp([num2str(pc), '% Complete']);
        
        nSubs_Total = sub_vector(sub_count);
        %nSubs = zeros(1,nConds);
        
        num_trials(trial_count, sub_count) = t;
        sample_size(trial_count, sub_count) = nSubs_Total;
        
        for between_level = 1:num_between_levels
            %number of subjects to simulate
            nSubs(between_level) = round(nSubs_Total*prefs.condition_allocation(between_level));
            outcome_samples{between_level} = randsample(outcomes, nSims*nSubs(between_level), 'true', mean(cFinal{between_level}));
        end
        
        % do condition comparisons
        %p = cell(1, nComps);
        %%%
        % do t-tests of interest
        %%%
        power_marker = zeros(nComps, nSims);
        for comp = 1:nComps
            
            %figure out if a within or between comparison
            if level_between(prefs.comps(comp,1)) == level_between(prefs.comps(comp,2)) %within comparison
                bl = level_between(prefs.comps(comp,1));
                wl1 = level_within(prefs.comps(comp,1));
                wl2 = level_within(prefs.comps(comp,2));
                diff_scores = cond_score{bl, wl1}(outcome_samples{bl}) - cond_score{bl, wl2}(outcome_samples{bl});
                diff_scores = reshape(diff_scores, nSubs(bl), nSims);
                dz_vect = mean(diff_scores)./std(diff_scores);
                [~,p] = ttest(diff_scores);
                power_marker(comp, :) = p < prefs.alpha & dz_vect > 0;
                
            else %between comparison
                bl1 = level_between(prefs.comps(comp,1));
                bl2 = level_between(prefs.comps(comp,2));
                wl1 = level_within(prefs.comps(comp,1));
                wl2 = level_within(prefs.comps(comp,2));
                
                c1 = reshape(cond_score{bl1, wl1}(outcome_samples{bl1}), nSubs(bl1), nSims);
                c2 = reshape(cond_score{bl2, wl2}(outcome_samples{bl2}), nSubs(bl2), nSims);
                ds_vect = (mean(c1) - mean(c2)) ./...
                    (((nSubs(bl1) - 1)*(std(c1).^2) + (nSubs(bl2) - 1)*(std(c2).^2))/(nSubs(bl1) + nSubs(bl2) - 2)).^.5;
                [~,p] = ttest2(c1,c2);
                power_marker(comp, :) = p < prefs.alpha & ds_vect > 0;
                
                
                power(trial_count, sub_count) = mean(all(power_marker, 1));
            end
            
            
            
        end
        %%%
        
        %anova part
        Y = [];
        S = [];
        WF = [];
        BF = [];
        sm = 1;
        
        for between_level = 1:num_between_levels
            
            for c = 1:num_within_levels
                Y = [Y;reshape(cond_score{between_level, c}(outcome_samples{between_level}), nSubs(between_level), nSims)];
                S = [S; (sm:sm+nSubs(between_level)-1)'];
                WF = [WF; repmat(c, nSubs(between_level), 1)];
                BF = [BF; repmat(between_level, nSubs(between_level), 1)];
            end
            sm = sm + nSubs(between_level);
        end
        
        stats = mixed_anova_matrix(Y,S,WF,BF);
        
        if prefs.sig_ME1
            power_marker(end+1,:) = stats.Pbs < prefs.alpha;
        end
        if prefs.sig_ME2
            power_marker(end+1,:) = stats.Pws < prefs.alpha;
        end
        if prefs.sig_int
            power_marker(end+1,:) = stats.Pint < prefs.alpha;
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
%power_results.dz_vect = dz_vect; %effect size vector for each design
power_results.sub_vector = sub_vector;
power_results.trial_vector = trial_vector;
power_results.prefs.figure_width = prefs.figure_width;
power_results.prefs.figure_height = prefs.figure_height;

%plot
figure(2)
clf

power = round(power, 2);

MakeHeatMap(power_results)

end

function power_results = PowerAnalysisMixed(prefs)
if sum(prefs.condition_allocation) ~= 1
    error('Condition allocation does not add up to 1 (100%). Use fractions if this is due to rounding error (e.g., use 1/3 instead of .33)')
elseif length(prefs.condition_allocation) ~= prefs.f1_num_levels
    error('Must have an allocation amount for each between-subjects level.')
end

%simulation info
nSims = prefs.nSims; %number of experiments to simulate
nPilotTrials = sum(prefs.data(:,1) == 1 & prefs.data(:,3) == 1 & prefs.data(:,4) == 1);

%number of subs for each between-subs factor level

num_between_levels = prefs.f1_num_levels;
num_within_levels =  prefs.f2_num_levels;

cond = 0;
for f1 = 1:num_between_levels
    for f2 = 1:num_within_levels
        cond = cond + 1;
        level_between(cond) = f1;
        level_within(cond) = f2;
    end
end


for c = 1:num_between_levels
    between_subs{c} = unique(prefs.sub_nums(prefs.data(:,3) == c))';
    nPilotSubs(c) = length(between_subs{c});
end

sub_vector = prefs.N_range; %number of subs per simulation
trial_vector = fliplr(prefs.trial_range); %number of trials per condition
nComps = size(prefs.comps, 1); %number of comparisons of interest

%preallocate
power = zeros(length(trial_vector), length(sub_vector));
sample_size = zeros(length(trial_vector), length(sub_vector));
num_trials = zeros(length(trial_vector), length(sub_vector));

%set up data to be easily sampled in simulations
%%%%
%organize data {between_cond}sub*trial*within_cond
pilot_data = cell(1, num_between_levels);
for f1 = 1:prefs.f1_num_levels
    pilot_data{f1} = nan(nPilotSubs(f1), nPilotTrials, prefs.f2_num_levels);
    for s = 1:nPilotSubs(f1)
        for f2 = 1:prefs.f2_num_levels
            pilot_data{f1}(s,:,f2) = prefs.data(prefs.data(:,1) == between_subs{f1}(s) & prefs.data(:,3) == f1 & prefs.data(:,4) == f2, 2)';
        end
    end
end


for trial_count= 1:length(trial_vector)
    
    t = trial_vector(trial_count);
    
    %figure out sampling variability based upon number of subjects
    for sub_count = 1:length(sub_vector)
        
        clc
        pc = round(100*((trial_count-1)*length(sub_vector) + sub_count - 1)...
            /(length(trial_vector)*length(sub_vector)));
        disp([num2str(pc), '% Complete']);
        
        nSubs_Total = sub_vector(sub_count);
        %nSubs = zeros(1,nConds);
        
        
        
        for f1 = 1:prefs.f1_num_levels
            %number of subjects to simulate
            nSubs_Total = sub_vector(sub_count);
            sample_size(trial_count, sub_count) = nSubs_Total;
            nSubs(f1) = round(nSubs_Total*prefs.condition_allocation(f1));
            num_trials(trial_count, sub_count) = t;
            
            total_num_trials = nSubs(f1)*nSims*t;
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
            
            for f2 = 1:prefs.f2_num_levels
                cond_scores{f1}{f2} = [];
            end
            
            for sim_round = 1:total_sim_rounds
                
                nSimsRound = simsPerRound(sim_round);
                
                %select random subjects
                sim_subs = randsample(1:nPilotSubs(f1), nSubs(f1)*nSimsRound, 'true');
                sim_subs = repelem(sim_subs, t)';
                
                %select random trials, for each condition, generate data
                trial_nums = zeros(length(sim_subs), prefs.f2_num_levels);
                for f2 = 1:prefs.f2_num_levels
                    trial_nums(:,f2) = ceil(rand(length(sim_subs), 1)*nPilotTrials);
                    cond_scores_tmp = pilot_data{f1}(sub2ind(size(pilot_data{f1}), sim_subs, trial_nums(:,f2), repmat(f2, length(sim_subs), 1)));
                    cond_scores{f1}{f2} = [cond_scores{f1}{f2}, squeeze(mean(permute(reshape(cond_scores_tmp, t, nSubs(f1), nSimsRound), [2 1 3]),2))];
                end
            end
        end
        
        num_trials(trial_count, sub_count) = t;
        sample_size(trial_count, sub_count) = nSubs_Total;
        
        % do condition comparisons
        %p = cell(1, nComps);
        %%%
        % do t-tests of interest
        %%%
        power_marker = zeros(nComps, nSims);
        for comp = 1:nComps
            %figure out if a within or between comparison
            if level_between(prefs.comps(comp,1)) == level_between(prefs.comps(comp,2)) %within comparison
                bl = level_between(prefs.comps(comp,1));
                wl1 = level_within(prefs.comps(comp,1));
                wl2 = level_within(prefs.comps(comp,2));
                diff_scores = cond_scores{bl}{wl1} - cond_scores{bl}{wl2};
                diff_scores = reshape(diff_scores, nSubs(bl), nSims);
                dz_vect = mean(diff_scores)./std(diff_scores);
                [~,p] = ttest(diff_scores);
                power_marker(comp, :) = p < prefs.alpha & dz_vect > 0;
                
            else %between comparison
                bl1 = level_between(prefs.comps(comp,1));
                bl2 = level_between(prefs.comps(comp,2));
                wl1 = level_within(prefs.comps(comp,1));
                wl2 = level_within(prefs.comps(comp,2));
                
                c1 = cond_scores{bl1}{wl1};
                c2 = cond_scores{bl2}{wl2};
                ds_vect = (mean(c1) - mean(c2)) ./...
                    (((nSubs(bl1) - 1)*(std(c1).^2) + (nSubs(bl2) - 1)*(std(c2).^2))/(nSubs(bl1) + nSubs(bl2) - 2)).^.5;
                [~,p] = ttest2(c1,c2);
                power_marker(comp, :) = p < prefs.alpha & ds_vect > 0;
                
                
                power(trial_count, sub_count) = mean(all(power_marker, 1));
            end
        end
        %%%
        
        
        %anova part
        Y = [];
        S = [];
        WF = [];
        BF = [];
        sm = 1;
        
        for between_level = 1:num_between_levels
            
            for c = 1:num_within_levels
                Y = [Y;reshape(cond_scores{between_level}{c}, nSubs(between_level), nSims)];
                S = [S; (sm:sm+nSubs(between_level)-1)'];
                WF = [WF; repmat(c, nSubs(between_level), 1)];
                BF = [BF; repmat(between_level, nSubs(between_level), 1)];
            end
            sm = sm + nSubs(between_level);
        end
        
        stats = mixed_anova_matrix(Y,S,WF,BF);
        
        if prefs.sig_ME1
            power_marker(end+1,:) = stats.Pbs < prefs.alpha;
        end
        if prefs.sig_ME2
            power_marker(end+1,:) = stats.Pws < prefs.alpha;
        end
        if prefs.sig_int
            power_marker(end+1,:) = stats.Pint < prefs.alpha;
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
%power_results.dz_vect = dz_vect; %effect size vector for each design
power_results.sub_vector = sub_vector;
power_results.trial_vector = trial_vector;
power_results.prefs.figure_width = prefs.figure_width;
power_results.prefs.figure_height = prefs.figure_height;

power = round(power, 2);

MakeHeatMap(power_results)

end

function MakeHeatMap(power_results)
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
elseif numel(power) == 1
    xlim([0,1])
    ylim([0,1])
    text(.2, .6,['Total Subjects = ', num2str(power_results.n)], 'FontSize', 20)
    text(.2, .5,['Trials Per Condition = ', num2str(power_results.num_trials)], 'FontSize', 20)
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

function stats = rm_anova2_matrix(Y,S,F1,F2)
%
% function stats = rm_anova2(Y,S,F1,F2,FACTNAMES)
%
% Two-factor, within-subject repeated measures ANOVA.
% For designs with two within-subject factors.
%
% Parameters:
%    Y          dependent variable (numeric) in a column vector
%    S          grouping variable for SUBJECT
%    F1         grouping variable for factor #1
%    F2         grouping variable for factor #2
%    FACTNAMES  a cell array w/ two char arrays: {'factor1', 'factor2'}
%
%    Y should be a 1-d column vector with all of your data (numeric).
%    The grouping variables should also be 1-d numeric, each with same
%    length as Y. Each entry in each of the grouping vectors indicates the
%    level # (or subject #) of the corresponding entry in Y.
%
% Returns:
%    stats is a cell array with the usual ANOVA table:
%      Source / ss / df / ms / F / p
%
% Notes:
%    Program does not do any input validation, so it is up to you to make
%    sure that you have passed in the parameters in the correct form:
%
%       Y, S, F1, and F2 must be numeric vectors all of the same length.
%
%       There must be at least one value in Y for each possible combination
%       of S, F1, and F2 (i.e. there must be at least one measurement per
%       subject per condition).
%
%       If there is more than one measurement per subject X condition, then
%       the program will take the mean of those measurements.
%
% Aaron Schurger (2005.02.04)
%   Derived from Keppel & Wickens (2004) "Design and Analysis" ch. 18
%

%
% Revision history...
%
% 11 December 2009 (Aaron Schurger)
%
% Fixed error under "bracket terms"
% was: expY = sum(Y.^2);
% now: expY = sum(sum(sum(MEANS.^2)));
%
% 05 April 2019 (Roger Strong)
% Can input multiple columns of Y at once (do separate ANOVA on each
% column, useful for simulations)
% Have a few slow steps (creating AS and BS with for loops)

F1_lvls = unique(F1);
F2_lvls = unique(F2);
Subjs = unique(S);

a = length(F1_lvls); % # of levels in factor 1
b = length(F2_lvls); % # of levels in factor 2
n = length(Subjs); % # of subjects

AB = zeros(a,b,size(Y,2));

for i = 1:a
    for j = 1:b
        AB(i, j, :) = sum(Y(F1 == i & F2 == j, :));
    end
end


%original
AS = zeros(a, n, size(Y,2));
for i = 1:a
    for sub = 1:n
        AS(i, sub, :) = sum(Y(S == sub & F1 == i, :));
    end
end
BS = zeros(b, n, size(Y,2));
for j = 1:b
    for sub = 1:n
        BS(j, sub, :) = sum(Y(S == sub & F2 == j, :));
    end
end

%%%tried without subject (n) for loop, even slower though
% %new
% Y2 = reshape(Y,numel(Y),1);
% S2 = repmat(S, size(Y,2), 1);
% F12 = repmat(F1, size(Y,2), 1);
% F22 = repmat(F2, size(Y,2), 1);
%
% AS2 = zeros(a, n, size(Y,2));
% for i = 1:a
%     tmp = Y2(F12 == i);
%     AS2(i, :, :) = sum(permute(reshape(tmp, n, a, size(Y,2)), [2 1 3]));
% end
% BS2 = zeros(b, n, size(Y,2));
% for i = 1:b
%     tmp = Y2(F22 == i);
%     BS2(i, :, :) = sum(permute(reshape(tmp, n, b, size(Y,2)), [2 1 3]));
% end


A = squeeze(sum(AB,2)); % sum across columns, so result is ax1 column vector
B = permute(squeeze(sum(AB,1)), [2 1]); % sum across rows, so result is 1xb row vector
S = permute(squeeze(sum(AS,1)), [2 1]); % sum across columns, so result is 1xs row vector
T = sum(A); % could sum either A or B or S, choice is arbitrary

% degrees of freedom
dfA = a-1;
dfB = b-1;
dfAB = (a-1)*(b-1);
dfS = n-1;
dfAS = (a-1)*(n-1);
dfBS = (b-1)*(n-1);
dfABS = (a-1)*(b-1)*(n-1);

% bracket terms (expected value)
expA = (sum(A.^2)./(b*n))';
expB = sum(B.^2, 2)./(a*n);
expAB = squeeze(sum(sum(AB.^2),2)./n);


expS = sum(S.^2, 2)./(a*b);
expAS = squeeze(sum(sum(AS.^2), 2)./b);
expBS = squeeze(sum(sum(BS.^2), 2)./a);
expY = sum(Y.^2)';
expT = (T.^2 ./ (a*b*n))';

% sums of squares
ssA = expA - expT;
ssB = expB - expT;
ssAB = expAB - expA - expB + expT;
ssS = expS - expT;
ssAS = expAS - expA - expS + expT;
ssBS = expBS - expB - expS + expT;
ssABS = expY - expAB - expAS - expBS + expA + expB + expS - expT;
ssTot = expY - expT;

% mean squares
msA = ssA ./ dfA;
msB = ssB ./ dfB;
msAB = ssAB ./ dfAB;
msS = ssS ./ dfS;
msAS = ssAS ./ dfAS;
msBS = ssBS ./ dfBS;
msABS = ssABS ./ dfABS;

% f statistic
fA = msA ./ msAS;
fB = msB ./ msBS;
fAB = msAB ./ msABS;

% p values
stats.pA = 1-fcdf(fA,dfA,dfAS);
stats.pB = 1-fcdf(fB,dfB,dfBS);
stats.pAB = 1-fcdf(fAB,dfAB,dfABS);

% return values
% stats = {'Source','SS','df','MS','F','p';...
%          FACTNAMES{1}, ssA, dfA, msA, fA, pA;...
%          FACTNAMES{2}, ssB, dfB, msB, fB, pB;...
%          [FACTNAMES{1} ' x ' FACTNAMES{2}], ssAB, dfAB, msAB, fAB, pAB;...
%          [FACTNAMES{1} ' x Subj'], ssAS, dfAS, msAS, [], [];...
%          [FACTNAMES{2} ' x Subj'], ssBS, dfBS, msBS, [], [];...
%          [FACTNAMES{1} ' x ' FACTNAMES{2} ' x Subj'], ssABS, dfABS, msABS, [], []};


end

function res=mixed_anova_matrix(Y,S,WF,BF)
% simple function for mixed (between- and within-subjects) ANOVA
%
% Based loosely on BWAOV2 (http://www.mathworks.com/matlabcentral/fileexchange/5579-bwaov2) by Antonio Trujillo-Ortiz
%  (insofar as I used that function to figure out the basic equations, as it is apparently very hard to find documentation
%  on mixed-model ANOVAs on the Internet). However, the code is all original.
%
% The major advantage of this function over the existing BWAOV2 is that it corrects a bug that occurs when the groups
%  have unequal numbers of subjects, as pointed out in the Matlab Central File Exchange by Jaewon Hwang. The code is also,
%  at least in my opinion, much cleaner.
%
% At present this function only supports mixed models with a single between-subjects factor and a single within-subjects
%  (repeated measures) factor, each with as many levels as you like. I would be happy to add more bells and whistles in
%  future editions, such as the ability to define multiple factors, apply Mauchly's test and add in non-sphericity
%  corrections when necessary, etc. I'm a better programmer than I am a statistician, though, so if anyone out there would
%  like to lend a hand with the math (e.g., feed me the equations for these features), I'd be more than happy to implement
%  them. Email matthew DOT r DOT johnson AT aya DOT yale DOT edu
%
% Also feel free to modify this file for your own purposes and upload your changes to the Matlab Central File Exchange if
%  you like.
%
% I have checked this function against the example data in David M. Lane's HyperStat online textbook, which is the same
%  data that breaks BWAOV2 (http://davidmlane.com/hyperstat/questions/Chapter_14.html, question 6). I have also checked it
%  against SPSS and gotten identical results. However, I haven't tested every possible case so bugs may remain. Use at
%  your own risk. If you find bugs and let me know, I'm happy to try to fix them.
%
% ===============
%      USAGE
% ===============
%
% Inputs:
%
% X: design matrix with four columns (future versions may allow different input configurations)
%     - first column  (i.e., X(:,1)) : all dependent variable values
%     - second column (i.e., X(:,2)) : between-subjects factor (e.g., subject group) level codes (ranging from 1:L where
%         L is the # of levels for the between-subjects factor)
%     - third column  (i.e., X(:,3)) : within-subjects factor (e.g., condition/task) level codes (ranging from 1:L where
%         L is the # of levels for the within-subjects factor)
%     - fourth column (i.e., X(:,4)) : subject codes (ranging from 1:N where N is the total number of subjects)
%
% suppress_output: defaults to 0 (meaning it displays the ANOVA table as output). If you don't want to display the table,
%  just pass in a non-zero value
%
% Outputs:
%
% SSQs, DFs, MSQs, Fs, Ps : Sum of squares, degrees of freedom, mean squares, F-values, P-values. All the same values
%  that are shown in the table if you choose to display it. All will be cell matrices. Values within will be in the same
%  order that they are shown in the output table.
%
% Enjoy! -MJ


% if nargin < 1,
%    error('No input');
% end;
%
% if nargin < 2 || isempty(suppress_output)
%     suppress_output=0;
% end

bs_levels=sort(unique(BF));
ws_levels=sort(unique(WF));
subj_levels=sort(unique(S));
n_bs_levels=length(bs_levels);
n_ws_levels=length(ws_levels);
n_subjects=length(subj_levels);


cell_totals = nan(n_bs_levels, n_ws_levels, size(Y,2));
for i=1:n_bs_levels
    for j=1:n_ws_levels
        this_cell_inds = BF==i & WF==j;
        n_subs_per_cell(i,j)= sum(this_cell_inds); %#ok<AGROW>
        cell_totals(i, j, :) = sum(Y(this_cell_inds, :)); %#ok<AGROW>
    end
end


for k=1:n_subjects
    subj_totals(k, :)= sum(Y(S==k, :)); %#ok<AGROW>
end

correction_term = sum(Y).^2 / size(Y,1);
SStot = sum(Y.^2) - correction_term;
%don't really need this for calculations, but can uncomment if we want to print
% DFtot = length(all_dvs) - 1; %total degrees of freedom

%subject "factor" (i.e. differences in subject means)
SSsub = sum(subj_totals .^ 2, 1)./n_ws_levels - correction_term;


%between-subjects factor
SStmp=[];
for i=1:n_bs_levels
    SStmp(i,:)= sum(cell_totals(i,:,:), 2).^2 ./ sum(n_subs_per_cell(i,:)); %#ok<AGROW>
end

SSbs    = sum(SStmp) - correction_term;

DFbs    = n_bs_levels - 1;
MSbs    = SSbs ./ DFbs;
%error terms for between-subjects factor
ERRbs   = SSsub - SSbs;
DFERRbs = n_subjects - n_bs_levels;
MSERRbs = ERRbs ./ DFERRbs;


%correction with harmonic mean of cell sizes if cell sizes are not all equal
n_subs_hm=harmmean(n_subs_per_cell(:));

cell_totals_hm = nan(n_bs_levels,n_ws_levels,size(Y,2));
for i=1:n_bs_levels
    for j=1:n_ws_levels
        cell_totals_hm(i,j,:) = cell_totals(i,j,:) ./ n_subs_per_cell(i,j) * n_subs_hm;
    end
end
correction_term_hm = squeeze(sum(sum(cell_totals_hm, 2))).^2 / (n_subs_hm * n_bs_levels * n_ws_levels);
n_subs_per_cell_hm = ones(n_bs_levels,n_ws_levels) * n_subs_hm;


%within-subjects factor
SStmp=[];
for j=1:n_ws_levels
    SStmp(j,:)=(squeeze(sum(cell_totals_hm(:,j,:), 1)).^2) ./ sum(n_subs_per_cell_hm(:,j)); %#ok<AGROW>
end

SSws  = sum(SStmp) - correction_term_hm';
DFws  = n_ws_levels - 1;
MSws  = SSws ./ DFws;

%uncorrected version of within-subjects factor for calculating interaction
SStmp=[];
for j=1:n_ws_levels
    SStmp(j, :)=(squeeze(sum(cell_totals(:,j,:), 1)).^2) ./ sum(n_subs_per_cell(:,j)); %#ok<AGROW>
end
SSws_unc = sum(SStmp) - correction_term;


n_subs_per_cell_rep = repmat(n_subs_per_cell, 1, 1, size(Y,2));
%interaction of between-subjects and within-subjects factor
SStmp = squeeze(sum(sum( (cell_totals.^2) ./ n_subs_per_cell_rep, 2)));


SSint = SStmp' - SSbs - SSws_unc - correction_term;
DFint = DFbs * DFws;
MSint = SSint ./ DFint;

%error terms (for both within-subjects factor and interaction)
ERRws   = SStot - SSbs - ERRbs - SSws_unc - SSint;
DFERRws = DFERRbs .* DFws;
MSERRws = ERRws ./ DFERRws;


%F-values
Fbs  = MSbs  ./ MSERRbs;
Fws  = MSws  ./ MSERRws;
Fint = MSint ./ MSERRws;

%P-values
res.Pbs  = 1-fcdf(Fbs, DFbs, DFERRbs);
res.Pws  = 1-fcdf(Fws, DFws, DFERRws);
res.Pint = 1-fcdf(Fint,DFint,DFERRws);


% SSQs = { SSbs; ERRbs;   SSws; SSint; ERRws   };
% DFs  = { DFbs; DFERRbs; DFws; DFint; DFERRws };
% MSQs = { MSbs; MSERRbs; MSws; MSint; MSERRws };
% Fs   = { Fbs;  [];      Fws;  Fint;  []      };
% Ps   = { Pbs;  [];      Pws;  Pint;  []      };
end
