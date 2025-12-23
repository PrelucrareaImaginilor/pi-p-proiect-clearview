% --Functie detectie glaucom--
function Compute_CDRs_ORIGA()
%CDR - Cup To Disc Ratio

%se incarca baza de date de comparatie
load('OrigaList.mat');

n = numel(Origa);

trainIdx = 1:256;       
testIdx  = 257:n; 

%variabile pentru tabelul de output
idx= (1:n)';
Filename= cell(n,1);

for i= 1:n
    Filename{i}= Origa(i).Filename;
end

CDR_area= nan(n,1); %CDR de suprafata
CDR_horiz= nan(n,1); %CDR orizontal 
CDR_vert = nan(n,1); %CDR vertical
IsGlaucoma= [Origa.Glaucoma]';


%% --Algoritm de generare masti prin reteaua neuronala unet--

% % conversie masti .mat in .png cu labels 0,1,2
% labelFolder = 'labels';
% if ~exist(labelFolder,'dir')
%     mkdir(labelFolder);
% end
% 
% imageFiles = cell(n,1);
% labelFiles = cell(n,1);
% 
% for i = 1:n
%     imageFiles{i} = fullfile('images', Origa(i).Filename);
% 
%     [~, name, ~] = fileparts(Origa(i).Filename);
%     load(fullfile('manual marking', [name '.mat']), 'mask');
% 
%     imwrite(uint8(mask), fullfile(labelFolder, [name '.png']));
%     labelFiles{i} = fullfile(labelFolder, [name '.png']);
% end
% 
% %% Datastores
% imds = imageDatastore(imageFiles);
% 
% classes  = ["background","disc","cup"];
% labelIDs = [0 1 2];
% pxds = pixelLabelDatastore(labelFiles, classes, labelIDs);
% 
% pximds = pixelLabelImageDatastore(imds, pxds);
% 
% %% Retea unet
% inputSize = [256 256 3];
% numClasses = 3;
% lgraph = unetLayers(inputSize, numClasses);
% 
% pximds = transform(pximds, @(x) { ...
%     imresize(x.inputImage{1}, inputSize(1:2)), ...
%     imresize(x.pixelLabelImage{1}, inputSize(1:2), 'nearest') ...
% });
% 
% %% Antrenare
% options = trainingOptions('adam', ...
%     'MaxEpochs', 3, ...
%     'MiniBatchSize', 8, ...
%     'Plots','training-progress', ...
%     'Verbose', true);
% 
% 
% net = trainNetwork(pximds, lgraph, options);
% save('U-Net_ORIGA.mat','net');
% 
% %% CDR
% CDR_area = zeros(n,1);
% 
% for i = 1:n
%     I = imread(imageFiles{i});
%     I = imresize(I, inputSize(1:2));
% 
%     C = semanticseg(I, net);
% 
%     cup  = (C == "cup");
%     disc = (C == "cup") | (C == "disc");
% 
%     CDR_area(i) = sum(cup(:)) / sum(disc(:));
% end
% 
% disp(CDR_area);

%% --Generarea mastilor noi folosind reteaua 'net'--

load('U-Net_ORIGA.mat','net');

startIdx = 257;
endIdx= min(startIdx + 255, n);

inputSize = [256 256 3];

CDR_area = nan(n,1);
CDR_vert= nan(n,1);
CDR_horiz= nan(n,1);
CupPixels = nan(n,1);
DiscPixels = nan(n,1);

% Generare CDR pe setul train pentru stabilirea pragurilor
for i = trainIdx
    I = imread(fullfile('images', Origa(i).Filename));
    I = imresize(I, inputSize(1:2));

    C = semanticseg(I, net);
    cupMask  = (C == "cup");
    discMask = (C == "cup") | (C == "disc");

    CDR_area(i)  = sum(cupMask(:)) / sum(discMask(:));
    CDR_horiz(i) = max(sum(cupMask,1)) / max(sum(discMask,1));
    CDR_vert(i)  = max(sum(cupMask,2)) / max(sum(discMask,2));
    fprintf("Iteratia: %g ", i );
end

for i = startIdx:endIdx
    I= imread(fullfile('images', Origa(i).Filename));
    I= imresize(I, inputSize(1:2));

    %segmentare 
    C= semanticseg(I, net);
    cupMask= (C == "cup");
    discMask= (C == "cup") | (C == "disc");

    CupPixels(i)= sum(cupMask(:));
    DiscPixels(i)= sum(discMask(:));
    if DiscPixels(i) > 0
        % CDR de suprafata
        CDR_area(i)= CupPixels(i) / DiscPixels(i);

        % CDR orizontal- max suma pe coloane
        CDR_horiz(i)= max(sum(cupMask,1)) / max(sum(discMask,1));

        % CDR vertical- max suma pe randuri
        CDR_vert(i)= max(sum(cupMask,2)) / max(sum(discMask,2));
    else
        CDR_area(i)= NaN;
        CDR_horiz(i)= NaN;
        CDR_vert(i)= NaN;
    end

    figure(1); clf;

    subplot(1,3,1);
    imshow(I);
    title(sprintf('Original (%d)', i));

    subplot(1,3,2);
    imshow(discMask);
    title('Disc');

    subplot(1,3,3);
    imshow(cupMask);
    title('Cup');

    drawnow;

end



%% --Algoritm de predictie IsGlaucoma--
%setup praguri
th= 0:0.01:1;
defaultThr = 0.6;  

%ponderi pentru fiecare CDR
w_area = 0.7; w_vert = 0.2; w_horiz = 0.1;

%stabilim praguri pe baza datelor de comparat
TrueLabel= IsGlaucoma;
HasTrueLabels = any(~isnan(TrueLabel));

% calcul praguri pentru CDRa,h,v cu ROC si Youden
if HasTrueLabels
    validA = ismember((1:n)', trainIdx) & ~isnan(CDR_area) & ~isnan(TrueLabel);
    if any(validA)
        [FPRa, TPRa] = GetROC(CDR_area(validA), TrueLabel(validA)); 
        %Youden's index: J=TPR-FPR=sensitivity + specificity − 1
        [~, ia]= max(TPRa - FPRa); %gaseste indexul unde J este maxim -> prag optim
        ThrArea= th(ia); %pragul optim pentru CDR_area extras din th
    else
        ThrArea= defaultThr;
        FPRa= []; TPRa= [];
    end
    %analog pentru CDRv si CDRh
    validV = ismember((1:n)', trainIdx) & ~isnan(CDR_vert) & ~isnan(TrueLabel);
    if any(validV)
        [FPRv, TPRv]= GetROC(CDR_vert(validV), TrueLabel(validV));
        [~, iv]= max(TPRv - FPRv);
        ThrVert= th(iv);
    else
        ThrVert= defaultThr;
        FPRv= []; TPRv= [];
    end

    validH = ismember((1:n)', trainIdx) & ~isnan(CDR_horiz) & ~isnan(TrueLabel);
    if any(validH)
        [FPRh, TPRh]= GetROC(CDR_horiz(validH), TrueLabel(validH));
        [~, ih]= max(TPRh - FPRh);
        ThrHoriz= th(ih);
    else
        ThrHoriz= defaultThr;
        FPRh=[]; TPRh= [];
    end
else
    ThrArea= defaultThr;
    ThrVert= defaultThr;
    ThrHoriz= defaultThr;
    FPRa= []; TPRa= []; FPRv= []; TPRv= []; FPRh= []; TPRh= [];
end

fprintf('Praguri: area=%.2f, vert=%.2f, horiz=%.2f\n', ThrArea, ThrVert, ThrHoriz);

% transformam scorurile (CDR) in valori binare
nEntries= numel(idx);
PredArea= false(nEntries,1);
PredVert= false(nEntries,1);
PredHoriz= false(nEntries,1);

PredArea(~isnan(CDR_area))= CDR_area(~isnan(CDR_area))>= ThrArea;
PredVert(~isnan(CDR_vert))= CDR_vert(~isnan(CDR_vert))>= ThrVert;
PredHoriz(~isnan(CDR_horiz))= CDR_horiz(~isnan(CDR_horiz))>= ThrHoriz;

% scor combinat
CombinedScore = nan(nEntries,1);
for i= 1:nEntries
    S= 0; %suma ponderata
    SW= 0; %suma ponderilor
    if ~isnan(CDR_area(i))
        S= S + w_area * CDR_area(i); %ponderea CDR area la suma
        SW= SW + w_area;  %se aduna ponderile
    end
    %analog
    if ~isnan(CDR_vert(i))
        S= S + w_vert * CDR_vert(i);  
        SW= SW + w_vert;  
    end
    if ~isnan(CDR_horiz(i)) 
        S= S + w_horiz * CDR_horiz(i); 
        SW= SW + w_horiz; 
    end
    if SW > 0
        CombinedScore(i)= S / SW; %media ponderata
    end
end

% prag scor combinat
if HasTrueLabels
    validS = ismember((1:n)', trainIdx) & ~isnan(CombinedScore) & ~isnan(TrueLabel);
    if any(validS)
        [FPRs, TPRs]=GetROC(CombinedScore(validS), TrueLabel(validS));
        [~, is]=max(TPRs - FPRs);
        ThrCombined=th(is);
    else
        ThrCombined=0.5;
        FPRs=[]; TPRs=[];
    end
else
    ThrCombined = 0.5;
    FPRs=[]; TPRs=[];
end

fprintf('Prag combinat = %.3f\n', ThrCombined);

% predictie finala
FinalPrediction = false(nEntries,1);
weights = [w_area, w_vert, w_horiz];

for i = 1:nEntries
    if ~isnan(CombinedScore(i))
        FinalPrediction(i)= CombinedScore(i)>= ThrCombined;
    else
        available= double([~isnan(CDR_area(i)), ~isnan(CDR_vert(i)), ~isnan(CDR_horiz(i))]);
        voteVals= double([PredArea(i), PredVert(i), PredHoriz(i)]); 
        wsum= sum(weights .* available); %suma ponderilor pentru metricele disponibile

        if wsum <= eps %daca nicio metrica nu este disponibila
            FinalPrediction(i)= false;
        else
            %determinam ponderea valorilor true
            weightedVote= sum(voteVals .* weights .* available);
            FinalPrediction(i)= weightedVote>= 0.5 * wsum;
        end
    end
end

%% Comparatie cu date OrigaList si afisare date
evalIdx = ismember((1:n)', testIdx) & ~isnan(TrueLabel);
if any(evalIdx)
    labs = TrueLabel(evalIdx);
    preds = FinalPrediction(evalIdx);
    TP= sum((labs==1)& (preds==1));
    FP= sum((labs==0)& (preds==1));
    TN= sum((labs==0)& (preds==0));
    FN= sum((labs==1)& (preds==0));
    Nval= TP+FP+TN+FN;
    accuracy= (TP + TN) / max(1, Nval);
    sensitivity= TP / max(1, (TP+FN));
    specificity= TN / max(1, (TN+FP));
    fprintf('Evaluare vs Origa: N=%d TP=%d FP=%d TN=%d FN=%d\n', Nval,TP,FP,TN,FN);
    fprintf('Acc=%.3f Sens=%.3f Spec=%.3f\n', accuracy, sensitivity, specificity);
else
    fprintf('Nu exista etichete disponibile pentru evaluare.\n');
end

%% Plot ROC
if HasTrueLabels
    figure(2); clf; hold on; grid on;
    legendEntries = {};
    if ~isempty(FPRa)
        plot(FPRa, TPRa, 'r', 'LineWidth', 1.5); legendEntries{end+1}='CDR Area'; 
    end
    if ~isempty(FPRh)
        plot(FPRh, TPRh, 'g', 'LineWidth', 1.5); legendEntries{end+1}='CDR Horiz'; 
    end
    if ~isempty(FPRv)
        plot(FPRv, TPRv, 'b', 'LineWidth', 1.5); legendEntries{end+1}='CDR Vert'; 
    end
    if ~isempty(FPRs)
        plot(FPRs, TPRs, 'k--', 'LineWidth', 1.5); legendEntries{end+1}='Combined'; 
    end
    if ~isempty(legendEntries)
        legend(legendEntries{:});
    end
    xlabel('FPR (1 - Specificity)');
    ylabel('Sensitivity (TPR)');
    title('ROC Curves (CDR metrics)');
    xlim([0 1]); ylim([0 1]);
    hold off;
end

% Realizare fisier CSV
PredNum = double(FinalPrediction);
TrueLabelnum = double(TrueLabel);

outT = table(idx, Filename, ...
    CDR_area, CDR_horiz, CDR_vert, TrueLabelnum, PredNum, CombinedScore, ...
    'VariableNames', {'Index','Filename', ...
    'CDR_area','CDR_horiz','CDR_vert','IsGlaucoma','PredictedGlaucoma','PredictedScore'});

writetable(outT, 'Output data.csv');

end

% --Functie validare output--
function [FPR, sens] = GetROC(scores, labels)
%ROC curb - Receiver Operating Characteristic curb

th = 0:0.01:1; %vector de praguri
nTh = length(th); %numar praguri
sens = nan(nTh,1); % sensitivity
spec = nan(nTh,1); % specifity
scores = scores(:); %valorile CDR
labels = labels(:);%IsGlaucoma

idx = 0;
for k = th %k=prag
    idx = idx + 1;
    p = scores >= k; %daca valoarea CDR >= prag -> 1(true), altfel 0(false)
    TP = sum((labels==1) & p); %true positive
    FP = sum((labels==0) & p); %false positive
    TN = sum((labels==0) & ~p); %true negative
    FN = sum((labels==1) & ~p); %false negative

    if TP + FN > 0 %proportia din cazurile reale pozitive care au fost detectate (sensitivity)
        sens(idx) = TP / (TP + FN);
    else
        sens(idx) = NaN;
    end

    if TN + FP > 0 %proportia din cazurile reale negative care au fost detectate (specificity)
        spec(idx) = TN / (TN + FP);
    else
        spec(idx) = NaN;
    end
end

FPR=1-spec; %false positive rate, axa X standard a ROC

end