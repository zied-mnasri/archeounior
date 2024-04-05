clear all
close all

%%% Load data %%%
origFolder = 'C:\Users\ziedm\Desktop\TempDocs\Research\Projects\UniOr\OmanDB\Oggetti_foto_cropped\';
imgFolder = 'C:\Users\ziedm\Desktop\TempDocs\Research\Projects\UniOr\Object_Detection_Classification\Data\Oggetti_foto_cropped\';

if exist(imgFolder,'dir')
      rmdir('C:\Users\ziedm\Desktop\TempDocs\Research\Projects\UniOr\Object_Detection_Classification\Data\Oggetti_foto_cropped\','s')%imgFolderTrain
end
mkdir ('C:\Users\ziedm\Desktop\TempDocs\Research\Projects\UniOr\Object_Detection_Classification\Data\Oggetti_foto_cropped\')
copyfile(origFolder,imgFolder)

LabelsFolder='C:\Users\ziedm\Desktop\TempDocs\Research\Projects\UniOr\Object_Detection_Classification\Data\';
imgData=readtable([LabelsFolder,'oggettiOmanAgg.csv']);


%%% Extract labels %%%
for ii=1:size(imgData,1)
    imgNames{ii}=[imgFolder,cell2mat(imgData{ii,1}),'jpg'];
end
imgLabels=imgData{:,4}
imgLabels = categorical(imgLabels);

%%% Build image datasotre %%%
imds = imageDatastore(imgFolder);
imgNb=length(imds.Files);
imds.Labels=(imgLabels(1:imgNb));

%%% Dispaly some samples %%%
figure;
perm = randperm(length(imds.Files),20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
    title(imds.Labels(perm(i)));
end

%%%% Compute the number of images for each class %%%%
labelCount = countEachLabel(imds)
labelNb=size(labelCount,1)

%%%% Specify the image size for the NN %%%%
img = readimage(imds,1);
sizeImg=size(img)
imageSize=[256 256 3];

%%%% Split data into training (80%) and validation (20%) %%%%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomize');

%%%% Reize all images to fit the NN input size %%%%
for ii=1:size(imdsTrain.Files,1)
    img=imread(imdsTrain.Files{ii});
    img=imresize(img,[imageSize(1),imageSize(2)]);
    delete (imdsTrain.Files{ii});
    imwrite(img,imdsTrain.Files{ii});
end
 
for ii=1:size(imdsValidation.Files,1)
    img=imread(imdsValidation.Files{ii});
    img=imresize(img,[imageSize(1),imageSize(2)]);
    delete (imdsValidation.Files{ii});
    imwrite(img,imdsValidation.Files{ii});
end

%%%% Data augmentation %%%%
augmenter = imageDataAugmenter( ...
    'RandRotation',[0 360], ...
    'RandScale',[0.5 1],...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

augmentedTrainingSet = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb','DataAugmentation', augmenter);

%%%% Define the NN architecture %%%%
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4);
    softmaxLayer
    classificationLayer];

%%%% Define the NN training options %%%%
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%%% NN model training %%%%   
net = trainNetwork(imdsTrain,layers,options);

%%%% NN model validation %%%%
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;


% Tabulate the results using a confusion matrix.
confMat = confusionmat(YValidation, YPred); % N target labels x N Predicted labels

% Convert confusion matrix into percentage form
confMatPer = bsxfun(@rdivide,confMat,sum(confMat,2));

%%%% Compute performance metrics %%%%
accuracy=trace(confMat)/sum(sum(confMat));

for ii=1:size(confMat,1)
    precision(ii)=confMat(ii,ii)/sum(confMat(:,ii));
    recall(ii)=confMat(ii,ii)/sum(confMat(ii,:));
    f1score(ii)=2*precision(ii)*recall(ii)/(precision(ii)+recall(ii));
end
display (['Accuracy = ',num2str(accuracy)])
display (['Precision = ',num2str(precision)])
display (['Recall = ',num2str(recall)])
display (['F1 score = ',num2str(f1score)])

%%%% Test on random samples %%%%
N=input(['Inserisci un numero di immagine tra 1 e ',num2str(length(imdsValidation.Files)),' :'])
imgN=readimage(imdsValidation,N);
imgNfileName=imdsValidation.Files{N};
YpredN=classify(net,imgN);

[pathstr,filename,ext] = fileparts(imgNfileName);
imgNfileNameOrig=[origFolder,filename,ext];
imgNOrig=imread(imgNfileNameOrig);

figure
subplot(121)
imshow(imgNOrig);
title(['Target: ',imdsValidation.Labels(N)]);

subplot(122)
imshow(imgN);
title(['Prediction: ',YpredN])