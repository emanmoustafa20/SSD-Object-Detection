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

%--------------------------------------------------------------------------------------
%Processing Video
%Create a video reader and writer
videoFileReader = VideoReader('car.avi');
myVideo = VideoWriter('myFile.avi');
%Create a  deployable video player
depVideoPlayer = vision.DeployableVideoPlayer;
open(myVideo);
%% Detect faces in each frame
while hasFrame(videoFileReader)
    
    % read video frame
    videoFrame = readFrame(videoFileReader);
    % process frame
    [bboxes,scores] = detect(detector,videoFrame, 'Threshold', 0.4);
    videoFrame = insertShape(videoFrame, 'Rectangle', bboxes);
    
    % Display video frame to screen
    depVideoPlayer(videoFrame);
    
    % Write frame to final video file
    writeVideo(myVideo, videoFrame);
    pause(1/videoFileReader.FrameRate);
    
end
close(myVideo)