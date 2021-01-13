%In the beginning, pretrained dataset is downloaded.
%It's downloaded to save time in training the data.
doTraining = false;
if ~doTraining && ~exist('ssdResNet50VehicleExample_20a.mat','file')
    disp('Downloading pretrained detector (44 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/ssdResNet50VehicleExample_20a.mat';
    websave('ssdResNet50VehicleExample_20a.mat',pretrainedURL);
end


    % Load pretrained detector for the example.
    pretrained = load('ssdResNet50VehicleExample_20a.mat');
    detector = pretrained.detector;


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
