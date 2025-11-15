% --Functie detectie glaucom--
function Compute_CDRs_ORIGA()
%CDR - Cup To Disc Ratio

%se incarca baza de date de comparatie
load('OrigaList.mat');

n = length(Origa);

%variabile pentru tabelul de output
idx = (1:n)';
Filename   = cell(n,1);
MaskExists = false(n,1);
CupPixels  = nan(n,1);
DiscPixels = nan(n,1);
CDR_area   = nan(n,1); %CDR de suprafata
CDR_horiz  = nan(n,1); %CDR orizontal
CDR_vert   = nan(n,1); %CDR vertical
IsGlaucoma = nan(n,1);

%iteram prin fiecare fisier
for i = 1:n
    Filename{i} = Origa(i).Filename;
    IsGlaucoma(i) = Origa(i).Glaucoma;
    maskFile = fullfile('manual marking', [Origa(i).Filename(1:end-4), '.mat']);
    %verificam daca exista fisierul masca pentru fiecare cele 650 imagini
    if exist(maskFile,'file')
        MaskExists(i) = true;
        S = load(maskFile,'mask');
        mask = S.mask;
        %valori masca -> 0- fundal, 1&2-disc(cuprinde cupa), 2-cupa
        cupMask  = (mask == 2);
        discMask = (mask >= 1);
        %pixeli totali in fiecare imagine
        CupPixels(i)  = sum(cupMask(:));
        DiscPixels(i) = sum(discMask(:));
        %verificam daca avem impartire la 0, daca masca este corect marcata
        if DiscPixels(i) > 0
            CDR_area(i)  = CupPixels(i) / DiscPixels(i); %CDR de suprafata
            CDR_horiz(i) = max(sum(cupMask,1)) / max(sum(discMask,1)); %sum(...,1) - returns a row vector of the sums of each column
            CDR_vert(i)  = max(sum(cupMask,2)) / max(sum(discMask,2)); %sum(...,2) -  returns a column vector of the sums of each row
        else
            CDR_area(i)  = NaN;
            CDR_horiz(i) = NaN;
            CDR_vert(i)  = NaN;
        end

        %caz in care nu exista masca
    else
        MaskExists(i) = false;
        CupPixels(i)  = NaN;
        DiscPixels(i) = NaN;
        CDR_area(i)   = NaN;
        CDR_horiz(i)  = NaN;
        CDR_vert(i)   = NaN;
    end
end
%verificam pentru fiecare masca daca diagnosticul corespunde cu baza de date originala
matches = [Origa.Glaucoma]' == IsGlaucoma; 
MatchYN = repmat("No", n,1);
MatchYN(matches) = "Yes";
%realizam tabelul cu toate datele
output_table= table(idx, Filename, CDR_area, CDR_horiz, CDR_vert, IsGlaucoma,MatchYN, ...
'VariableNames', {'Index','Filename','CDR_area','CDR_horiz','CDR_vert','IsGlaucoma', 'OriginalMatch'});
%tabelul este pus intr-un fisier extern
writetable(output_table, 'Output data.csv');

%filtrare NaN pentru fiecare score
validA = ~isnan(CDR_area) & ~isnan(IsGlaucoma);
validH = ~isnan(CDR_horiz) & ~isnan(IsGlaucoma);
validV = ~isnan(CDR_vert) & ~isnan(IsGlaucoma);

%apelare GetROC pentru fiecare CDR
[FPRa, TPRa] = GetROC(CDR_area(validA), IsGlaucoma(validA));
[FPRh, TPRh] = GetROC(CDR_horiz(validH), IsGlaucoma(validH));
[FPRv, TPRv] = GetROC(CDR_vert(validV), IsGlaucoma(validV));

%afisare grafice CDR in acelasi plot
figure; hold on; grid on;
plot(FPRa, TPRa, 'r', 'LineWidth', 1.5);
plot(FPRh, TPRh, 'g', 'LineWidth', 1.5);
plot(FPRv, TPRv, 'b', 'LineWidth', 1.5);
xlabel('FPR (1 - Specificity)');
ylabel('Sensitivity (TPR)');
title('ROC Curves');
legend('CDR Area','CDR Horiz','CDR Vert');
xlim([0 1]); ylim([0 1]);
hold off;

end
% --Functie validare output--
function [FPR, sens] =  GetROC(scores, labels)
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
