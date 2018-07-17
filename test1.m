% Auto Encoder using MatConvNet
% Parameters
N=100; % hidden blocks
% either absolute size or compression rate are accepted
LR=0.001; % learning rate
n_epochs=10; % epochs
bsize=100; % batchsize

% initialization
run('d:\matconvnet\matlab\vl_setupnn');
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.expDir = 'd:\codes\matlab\aetest\test1' ;

% load data
imdb = load('d:\codes\matlab\aetest\imdb.mat') ;
L=size(imdb.images.data,1);
S=size(imdb.images.data,3);
if (S>L)
    imdb.images.data=resize(imdb.images.data,L,L,1,[]);
end
S=size(imdb.images.data,4);
E=mean(reshape(imdb.images.data,L*L,[]));
V=std(reshape(imdb.images.data,L*L,[]));
E=reshape(E,1,1,1,[]);
V=reshape(V,1,1,1,[]);
imdb.images.data=(imdb.images.data-E)./V;
if (N<1)
    N=ceil(N*L*L);
end

% init network
rng('default');
rng(0) ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{(1/L/L)*randn(L,L,1,N, 'single'), zeros(1, N, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
%net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{(1/N)*randn(1,1,N,L*L, 'single'),zeros(1,L*L,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'sqloss') ;
% Meta parameters
net.meta.inputSize = [L L 1] ;
net.meta.trainOpts.learningRate = LR ;
net.meta.trainOpts.numEpochs = n_epochs ;
net.meta.trainOpts.batchSize = bsize ;
% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% train
[net, info] = cnn_train(net, imdb, @getBatch, ...
  'batchSize', net.meta.trainOpts.batchSize, ...
  'numEpochs', net.meta.trainOpts.numEpochs, ...
  'expDir', opts.expDir, 'errorFunction', 'square') ;
% @(x,y) getBatch(x,y)

% save the net
save([opts.expDir,'\aenet.mat'], '-struct', 'net') ;

% test the net
A=imdb.images.data(:,:,:,ceil(rand*S));
net.layers(end) = [] ;
res = vl_simplenn(net, A) ;
B=reshape(res(end).x,L,L);
figure;
subplot(1,2,1)
imshow((A+1)/2);
subplot(1,2,2)
imshow((B+1)/2);

% --------------------------------------------------------------------
function [images, labels] = getBatch(imdb, batch)
    images = imdb.images.data(:,:,:,batch) ;
    L=size(images,1);
    labels = reshape(images,1,1,L*L,[]);
end
% --------------------------------------------------------------------