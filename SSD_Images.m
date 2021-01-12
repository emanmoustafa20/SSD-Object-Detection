%In the beginning, pretrained dataset is downloaded.
%It's downloaded to save time in training the data.
doTraining = false;
if ~doTraining && ~exist('ssdResNet50VehicleExample_20a.mat','file')
    disp('Downloading pretrained detector (44 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/ssdResNet50VehicleExample_20a.mat';
    websave('ssdResNet50VehicleExample_20a.mat',pretrainedURL);
end

%Loading the dataset
unzip vehicle_zip.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

%Split data set to training set and test set
% 70% is selected to training and the rest for evaluation
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.7 * length(shuffledIndices) );
trainingData = vehicleDataset(shuffledIndices(1:idx),:);
testData = vehicleDataset(shuffledIndices(idx+1:end),:);

%Loading images and label data
imdsTrain = imageDatastore(trainingData{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingData(:,'vehicle'));
imdsTest = imageDatastore(testData{:,'imageFilename'});
bldsTest = boxLabelDatastore(testData(:,'vehicle'));

%Combine image and box label datastores.
trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest, bldsTest);

%NOW, Creating SSD Object Detection Network
%size of the training image
inputSize = [300 300 3];
%Number of object classes to detect
numClasses = width(vehicleDataset)-1;
%SSD layer
lgraph = ssdLayers(inputSize, numClasses, 'resnet50');

%Now, Data augmentation concept is applied to increase accuracy
%by Randomly flipping the image and associated box labels horizontally
%and Randomly scale the image, associated box labels.
augmentedTrainingData = transform(trainingData,@augmentData);

%Preprocess the augmented training data to prepare for training
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);

%Train SSD Object Detector
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ....
    'InitialLearnRate',1e-1, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.8, ...
    'MaxEpochs', 300, ...
    'VerboseFrequency', 50, ...
    'CheckpointPath', tempdir, ...
    'Shuffle','every-epoch');
if doTraining
    % Train the SSD detector.
    [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('ssdResNet50VehicleExample_20a.mat');
    detector = pretrained.detector;
end
%---------------------------------------------------------------------------------------
%As a QUICK TEST, YOU MAY UNCOMMENT FOLLOWING LINES& APPLY detector on one test image
data = read(testData);
I = data{1,1};
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I, 'Threshold', 0.4);
%Display
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)


%--------------------------------------------------------------------------
%Eventually, YOU MAY UNCOMMENT THE FOLLOWING LINES AND
%evaluate detector using test set
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(detector, preprocessedTestData, 'Threshold', 0.4);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

