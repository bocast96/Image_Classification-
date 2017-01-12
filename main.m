%% tools setup
%run /home/bocast96/vlfeat-0.9.20/toolbox/vl_setup
%run /home/bocast96/P3/Code/matconvnet/matlab/vl_setupnn
%run /home/bocast96/P3/Code/matconvnet/SetupMatConvNet

%% setup
clear;
clc;

%% Part 1 
%load & process data
k = 100;
db = load('/home/bocast96/P3/Code/matconvnet/data/cifar/cifar-10-batches-mat/data_batch_1.mat');
labels = db.labels;
data = db.data;

db = load('/home/bocast96/P3/Code/matconvnet/data/cifar/cifar-10-batches-mat/data_batch_2.mat');
labels = [labels; db.labels];
data = [data; db.data];

db = load('/home/bocast96/P3/Code/matconvnet/data/cifar/cifar-10-batches-mat/data_batch_3.mat');
labels = [labels; db.labels];
data = [data; db.data];

db = load('/home/bocast96/P3/Code/matconvnet/data/cifar/cifar-10-batches-mat/data_batch_4.mat');
labels = [labels; db.labels];
data = [data; db.data];

db = load('/home/bocast96/P3/Code/matconvnet/data/cifar/cifar-10-batches-mat/data_batch_5.mat');
labels = [labels; db.labels];
data = [data; db.data];

db = load('/home/bocast96/P3/Code/matconvnet/data/cifar/cifar-10-batches-mat/test_batch.mat');
testData = db.data;

%%
imgCount = length(data);
imgs = cell(1,imgCount);

for i = 1:imgCount
    Vec = data(i,:);
    imgs{1,i} = single((im2double(permute(reshape(Vec,32,32,3),[2,1,3])))); 
end

%%
testCount = length(testData);
testImgs = cell(1,testCount);

for i = 1:testCount
    Vec = data(i,:);
    testImgs{1,i} = single((im2double(permute(reshape(Vec,32,32,3),[2,1,3])))); 
end

%% Processing
[nnHist, leHist, clusters] = processing(imgs, k);

%% training models
nnMdl = fitcecoc(nnHist, labels);
leMdl = fitcecoc(leHist, labels);

%% testing
% processing test images
[nnHist, leHist, ~] = processing(testImgs, k, clusters);

%% testing different models
result1 = predict(nnMdl, nnHist);
result2 = predict(leMdl, leHist);

%% writing files
id = fopen('BoVW_nearest_neighbor_results.txt','w');
fprintf(id, 'Name: Boris Castillo\nLabels:');
fprintf(id, '%d', result1);
fclose(id);

id = fopen('BoVW_local_encoding_results.txt', 'w');
fprintf(id, 'Name: Boris Castillo\nLabels:');
fprintf(id, '%d', result2);
fclose(id);

%% Part 2 & 3
net = load('/home/bocast96/P3/Code/matconvnet/examples/imagenet/imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;
feats = zeros(imgCount, 4096);
%%
for i = 1:imgCount
   im = imgs{i};
   im = imresize(im, net.meta.normalization.imageSize(1:2));
   im = im - net.meta.normalization.averageImage;
   
   res = vl_simplenn(net, im);
   feats(i,:) = res(20).x(1,1,:);
end
%%
cnnMdl = fitcecoc(feats, labels);

testFeats = zeros(testCount, 4096);
%%
for i = 1:testCount
   im = testImgs{i};
   im = imresize(im, net.meta.normalization.imageSize(1:2));
   im = im - net.meta.normalization.averageImage;
   
   res = vl_simplenn(net, im);
   testFeats(i,:) = res(20).x(1,1,:);
end

cnnResult = predict(cnnMdl, testFeats);

%% saving result file
fopen('part2_results.txt', 'w');
fprintf(id, 'Name: Boris Castillo\nLabels:');
fprintf(id, '%d', cnnResult);
fclose(id);




