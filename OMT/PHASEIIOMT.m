clear; clc; close all;
addpath(genpath(fullfile('external')));
addpath(genpath(fullfile('function')));
% addpath(genpath(fullfile('wanghan')));

%% Load Cube
% Load Cube and Raw for making tensor
Cube = load(fullfile('maps', 'Cube', 'Cube_128_OMT.mat'));

%% Setting
P1_DensityName  = 'Exp-HE-Flair';
P1_DensityParam = 1.75;

P2_DensityName  = 'Exp-HE-Flair';
P2_DensityParam = 1.75;
DilateN = 5;
ConvN   = 5;

Homotopy_p = 11;
hfun_rho1 = 0.016;
Refine    = false;

%% Construct Tensor struct in make tensor step
[Y, X, Z] = meshgrid(1:128);
Tensor = Cube;
Tensor.Voxel = [X(:) Y(:) Z(:)] + 0.5;
clear X Y Z;

n = 128;
CubeSize = [n,n,n];
[i, j, l] = ind2sub(CubeSize, 1:n^3);
CubeV = sub2ind(flip(CubeSize), l, j, i);
[~, IC] = sort(CubeV);

Cube.V      = Tet.VolumeNormalize(Cube.T, Cube.V, 4*pi/3);
Cube.Bdry.V = Cube.V(Cube.VB, :);

%% Loop
path = fullfile('E:/GulouData/给军总/');
ImgLoc = fullfile(path, '给军总_NIFTI');
SaveLoc = fullfile(path, 'JunZongOMT');
PredLoc = fullfile(path, 'JunZongPred');
PatientList = dir(fullfile(PredLoc));

IdxLoc = fullfile(SaveLoc,'idx_PYTHON');
OMTLoc = fullfile(SaveLoc,'image');
MeshLoc = fullfile('GulouOMTPHASEI','mesh');

if ~exist(fullfile(IdxLoc),'dir')
    mkdir(fullfile(IdxLoc));
end
if ~exist(fullfile(OMTLoc),'dir')
    mkdir(fullfile(OMTLoc));
end

for k = 3 %: numel(PatientList)
    %% Read data
    PatientName = PatientList(k).name;
    Total_time = tic;
    fprintf('Read raw data\n');
    fprintf([PatientName '\n']);
    
    RawImg = niftiread(fullfile(ImgLoc, [PatientName '.nii.gz']));
    RawImg = RawImg + abs(min(RawImg(:)));
    mask  = niftiread(fullfile(PredLoc, PatientName, [PatientName '_WT.nii.gz']));
    mask  = mask > 0;
    Info = niftiinfo(fullfile(ImgLoc, [PatientName '.nii.gz']));
    
    %% Make mesh
    fprintf('Load mesh\n');
    load(fullfile(MeshLoc, [PatientName '.mat']));
    
    %% Phase 2 OMT with density control
    tic;
    fprintf('Phase2 OMT\n');
    % Define Mass(a.k.a Weight) (on T, V, Bdry.F and Bdry.V)
    DImg = Density.GenerateDImg(RawImg, mask, P2_DensityName, ...
        P2_DensityParam, DilateN, ConvN);
    % Define density (Gray)
    [Gray, Bdry] = Img2Gray(DImg, T, V, Bdry, VB);
    % Centralize and Mass Normalize
    [V, ~, ~] = Tet.MassNormalize(T, V, Gray);
    Bdry.V = V(VB,:);
    % Define mass (Weight)
    [Weight, Bdry] = DefineWeight(T, V, Bdry, Gray);
    % Mass Centralize
    MassCenter = sum(Weight.V .* V, 1) / sum(Weight.V);
    V = V - MassCenter;
    Bdry.V = V(VB,:);
    % OMT
    [S, Bdry] = Sphere.VOMT_CubeHomotopy(T, V, Bdry, VB, VI,...
        Weight, Cube, Homotopy_p);
    % Transform S to [1,129]^3
    S = S ./ ((4*pi/3)^(1/3) / 128);
    S = S - min(S) + 1;
    fprintf('Time Elapsed (Phase2 OMT): %f(s)\n', toc);
    %% Phase 2 Make Tensor
    load(fullfile(MeshLoc, [PatientName '.mat']), 'V'); % add 1018
    tic;
    fprintf('Phase2 Make tensor\n');
    [nx, ny, nz, ~]  = size(RawImg);
    fprintf('Phase2 Make Raw\n');
    Raw.V = MakeRaw(nx, ny, nz);
    % make InvIdx
    fprintf('Phase2 Make InvIdx\n');
    Tensor.V = InverseMap(T, V, S, Tensor.Voxel);
%     Tensor.V = Tensor.V * VolumeFactor + VolumeCenter;
%     InvIdx = knnsearch(Raw.V, Tensor.V);
    InvIdx = ksearch(Raw.V, Tensor.V, @(X) X + 0.5);
    % make Idx
    fprintf('Phase2 Make Idx\n');
    BrainIdx = ListBrainIdx(RawImg);
    Idx  = knnsearch(Tensor.V, Raw.V(BrainIdx,:));
    % make CubeImg
    CubeImg = MakeCubeImg(RawImg, InvIdx, 128);
    fprintf('Time Elapsed (Phase2 MakeTensor): %f(s)\n', toc);
    
    %% Save file
    save(fullfile(IdxLoc, [PatientName '.mat']), 'Idx', 'InvIdx');
    BraTS_Writer(Info, CubeImg, fullfile(OMTLoc, PatientName), true);
    fprintf('Total Elapsed Time: %f(s)\n', toc(Total_time));
    
    
end
