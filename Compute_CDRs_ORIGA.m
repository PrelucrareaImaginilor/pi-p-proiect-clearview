function Compute_CDRs_ORIGA()
%CDR - Cup To Disc Ratio

load('OrigaList.mat');

nImages = length(Origa);

CDR_manual  = zeros(nImages,1); % CDR calculat de specialisti
CDR_horiz   = zeros(nImages,1); % CDR orizontal calculat cu masca
CDR_area    = zeros(nImages,1); % aria cdr calculata cu masca
CDR_vert    = zeros(nImages,1); % CDR vertical calculat cu masca
isGlaucoma  = zeros(nImages,1);

for i = 1:nImages
    CDR_manual(i) = Origa(i).ExpCDR;
    isGlaucoma(i) = Origa(i).Glaucoma; 

    maskFile = fullfile('manual marking', [Origa(i).Filename(1:end-4), '.mat']);
    load(maskFile, 'mask');
    
    cupMask  = (mask == 2);
    discMask = (mask >= 1);
    
    CDR_horiz(i) = max(sum(cupMask,1)) / max(sum(discMask,1))
    CDR_area(i)  = sum(cupMask(:)) / sum(discMask(:))
    CDR_vert(i)  = max(sum(cupMask,2)) / max(sum(discMask,2))

end
    [seH, spH, acH] = GetROC(CDR_horiz, isGlaucoma);
    [seA, spA, acA] = GetROC(CDR_area, isGlaucoma);
    [seV, spV, acV] = GetROC(CDR_vert, isGlaucoma);

end

function [sens, spec, acc] = GetROC(scores, labels)
%ROC curb - Receiver Operating Characteristic curb

th = 0:0.01:1;
nTh = length(th);
sens = nan(nTh,1); % sensitivity 
spec = nan(nTh,1); % specifity
acc  = nan(nTh,1); % accuracy

scores = scores(:);
labels = labels(:);

idx = 0;
for k = th
    idx = idx + 1;
    p = scores >= k;
    TP = sum((labels==1) & p); %true positive
    FP = sum((labels==0) & p); %false positive
    TN = sum((labels==0) & ~p); %true negative
    FN = sum((labels==1) & ~p); %false negative

    if TP + FN > 0
        sens(idx) = TP / (TP + FN);
    else
        sens(idx) = NaN;
    end

    if TN + FP > 0
        spec(idx) = TN / (TN + FP);
    else
        spec(idx) = NaN;
    end

    tot = TP + FP + TN + FN;
    if tot > 0
        acc(idx) = (TP + TN) / tot;
    else
        acc(idx) = NaN;
    end
end

FPR = 1 - spec;
figure;
plot(FPR, sens, 'LineWidth', 1.5);
xlabel('FPR (1 - Specificity)');
ylabel('Sensitivity (TPR)');
title('ROC Curve');
xlim([0 1]); ylim([0 1]); grid on;

end
