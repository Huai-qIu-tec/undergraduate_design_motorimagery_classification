function data = getData(i)

subject_index = i; % 1-9

%% T data
session_type = 'T'; % T and E
dir_1 = ['E:\undergraduate design\数据集\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf']; % set your path of the downloaded data
[s, HDR] = mexSLOAD(dir_1);

% Label 
labeldir_1 = ['E:\undergraduate design\数据集\BCICIV_2a_gdf\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir_1);
label_1 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS; % use POS to get trials
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_1 = zeros(1000, 22, 288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_1(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end
data_1 = permute(data_1, [3, 2, 1]);
% 填充Nan
data_1(isnan(data_1)) = 0;


% E data
session_type = 'E';
dir_2 = ['E:\undergraduate design\数据集\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf'];
[s, HDR] = mexSLOAD(dir_2);

% Label 
labeldir_2 = ['E:\undergraduate design\数据集\BCICIV_2a_gdf\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir_2);
label_2 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_2 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_2(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end
data_2 = permute(data_2, [3, 2, 1]);

data_2(isnan(data_2)) = 0;

%% preprocessing
% option - band-pass filter
fc = 250; % sampling rate
Wl = 4; Wh = 40; % pass band
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:288
    data_1(:,:,j) = filtfilt(b,a,data_1(:,:,j));
    data_2(:,:,j) = filtfilt(b,a,data_2(:,:,j));
end

% option - a simple standardization

%% Save the data to a mat file 
data = data_1;
label = label_1;
% label = t_label + 1;
saveDir = ['E:\undergraduate design\数据集\BCICIV_2a_mat\A0',num2str(subject_index),'T.mat'];
save(saveDir,'data','label');

data = data_2;
label = label_2;
saveDir = ['E:\undergraduate design\数据集\BCICIV_2a_mat\A0',num2str(subject_index),'E.mat'];
save(saveDir,'data','label');

end


        
