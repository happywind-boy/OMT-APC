clear; 
clc; 
close all;
addpath(genpath(fullfile('external')));
addpath(genpath(fullfile('function')));

%% Load Cube
% Load Cube and Raw for making tensor
Cube = load('Cube_128_OMT.mat');

%% Setting
P1_DensityName  = 'Exp-HE-Flair';
P1_DensityParam = 1.75;

P2_DensityName  = 'Exp-HE-Flair';
P2_DensityParam = 1.75;
DilateN = 0;
ConvN   = 0;

Homotopy_p = 7;
hfun_rho1  = 0.016;
ratio      = 0.75;
Refine     = false;

%% Construct Tensor struct in make tensor step
[Y, X, Z] = meshgrid(1:128);
Tensor = Cube;
Tensor.Voxel = [X(:) Y(:) Z(:)] + 0.5;
clear X Y Z;

Cube.V      = Tet.VolumeNormalize(Cube.T, Cube.V, 4*pi/3);
Cube.Bdry.V = Cube.V(Cube.VB, :);

n = 128;
CubeSize = [n,n,n];
[i, j, l] = ind2sub(CubeSize, 1:n^3);
CubeV = sub2ind(flip(CubeSize), l, j, i);
[~, IC] = sort(CubeV);

%% Loop
path = fullfile('');
SaveLoc = fullfile(path, '');
BrainLoc = fullfile(path, '');
LabelLoc = fullfile(path, '');

IdxLoc = fullfile(SaveLoc,'');
OMTLoc = fullfile(SaveLoc,'');
LabLoc = fullfile(SaveLoc,'');
MeshLoc = fullfile(SaveLoc,'');
RegisLoc = fullfile(SaveLoc,'');
PatientList = dir(fullfile(BrainLoc));

if ~exist(RegisLoc,'dir')
    mkdir(RegisLoc);
end

if ~exist(IdxLoc,'dir')
    mkdir(IdxLoc);
end
if ~exist(OMTLoc,'dir')
    mkdir(OMTLoc);
end

if ~exist(MeshLoc,'dir')
    mkdir(MeshLoc);
end

if ~exist(LabLoc,'dir')
    mkdir(LabLoc);
end

for k = 3:numel(PatientList)
    %% Read data
    PatientFile = PatientList(k).name;
    PatientName = PatientFile;
    Total_time = tic;
    fprintf('Read raw data \n');
    fprintf([PatientName '\n']);
    
    ADCloc = fullfile(BrainLoc, PatientFile, [PatientName '_ADC.nii.gz']);
    
    if exist(ADCloc, 'file')
        
        if ~exist(fullfile(RegisLoc, PatientName),'dir')
            mkdir(fullfile(RegisLoc, PatientName));
        end
        
        if ~exist(fullfile(OMTLoc, PatientName),'dir')
            mkdir(fullfile(OMTLoc, PatientName));
        end
        
        T2_loc = fullfile(BrainLoc, PatientFile, [PatientName '_T2WI.nii.gz']);
        
        if exist(T2_loc, 'file')
            Info = niftiinfo(fullfile(BrainLoc, PatientFile, [PatientName '_T2WI.nii.gz']));
        else
            Info = niftiinfo(fullfile(BrainLoc, PatientFile, [PatientName '_FLAIR.nii.gz']));
        end
        
        T2Img = niftiread(Info);
        T1celoc = fullfile(BrainLoc, PatientFile, [PatientName '_T1CE.nii.gz']);
        T1ceImg = niftiread(T1celoc);
        T1wloc = fullfile(BrainLoc, PatientFile, [PatientName '_T1WI.nii.gz']);
        T1wImg = niftiread(T1wloc);
        Flairloc = fullfile(BrainLoc, PatientFile, [PatientName '_FLAIR.nii.gz']);
        FlairImg = niftiread(Flairloc);
        
        
        regisImg = cat(4, T1wImg, T1ceImg, FlairImg, T2Img);
        regisImg = double(regisImg);
        
        BraTS_Writer(Info, regisImg, fullfile(RegisLoc, PatientName, PatientName), true);
        DenImg = T2Img;
        RawImg = regisImg;
        
        %% Make mesh
        tic;
        fprintf('Make mesh\n');
        Bin = FindBrain(RawImg);
        Bin = BinAddSphere(Bin);
        Bin = imdilate(Bin, strel('cube', 3));
        [Bdry.F, Bdry.V] = MakeMesh_surface(Bin);
        DImg = ones(size(RawImg, 1, 2, 3));
        [T, V, Bdry, VB, VI] = MakeMesh_volume(Bdry, PatientName, ...
            DImg, hfun_rho1, Refine);
        fprintf('Time Elapsed (Make Mesh): %f(s)\n', toc);
        save(fullfile(MeshLoc, [PatientName '.mat']), 'T', 'V', 'Bdry',...
            'VB', 'VI');
        
        %% Phase1 OMT
        tic;
        fprintf('Phase1 OMT\n');
        % Define Mass(a.k.a Weight) (on T, V, Bdry.F and Bdry.V)
        DImg = Density.GenerateDImg(DenImg, [], P1_DensityName, P1_DensityParam,...
            DilateN, ConvN);
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
        
        S = S ./ ((4*pi/3)^(1/3) / 128);
        S = S - min(S) + 1;
        fprintf('Time Elapsed (Phase1 OMT): %f(s)\n', toc);
        
        %% Phase1 Make Tensor
        load(fullfile(MeshLoc, [PatientName '.mat']), 'V'); 

        tic;
        fprintf('Phase1 Make Tensor\n');
        [nx, ny, nz, ~]  = size(RawImg);
        fprintf('Phase1 Make Raw\n');
        Raw.V = MakeRaw(nx, ny, nz);
        % make InvIdx
        fprintf('Phase1 Make InvIdx\n');
        Tensor.V = InverseMap(T, V, S, Tensor.Voxel);
        InvIdx = ksearch(Raw.V, Tensor.V, @(X) X + 0.5);
        InvIdx_Raw = InvIdx;
        % make Idx
        fprintf('Phase1 Make Idx\n');
        BrainIdx = ListBrainIdx(RawImg);
        Idx = knnsearch(Tensor.V, Raw.V(BrainIdx,:));
        % make CubeImg
        if ~isempty(Lab)
            CubeLab = MakeCubeImg(Lab, InvIdx_Raw, 128);
        end
        CubeImg = MakeCubeImg(regisImg, InvIdx_Raw, 128);
        CubeADC = MakeCubeImg(ADCImg, InvIdx_Raw, 128);
        fprintf('Time Elapsed (Phase1 MakeTensor): %f(s)\n', toc);
        
        %% Save file
        save(fullfile(IdxLoc, [PatientName '.mat']), 'Idx', 'InvIdx', 'InvIdx_Raw');
        BraTS_Writer(Info, CubeImg, fullfile(OMTLoc, PatientName, PatientName), true);
        BraTS_Writer(Info, CubeADC, fullfile(OMTLoc, PatientName, [PatientName '_ADC']), true);
        fprintf('Total Elapsed Time: %f(s)\n', toc(Total_time));
        
   end
end
