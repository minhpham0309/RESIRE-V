%% Gradient Descent
%addpath('./source')
addpath( [pwd, '\src\'] );
%addpath( 'C:\Users\phmin\OneDrive\Desktop\Github\vector_tomo-master\GENFIRE-MATLAB' );
%% inputs
%projections = importdata('data/proj_lattice_100_d.mat');
%angles = importdata('data/ang_set_lattice_100_d.mat');

% projections = importdata('data/proj_gf_cosmic_rpie_hires1_sub5.mat');
% angles  = importdata('data/ang_cosmic_rpie_hires1_sub5.mat');
%support = importdata('data/support_hires.mat'); 
%support = importdata('data/support_lattice_100.mat'); 
% 
%projections  = importdata('capped_ML_vec_interp3_aligned.mat');
%angles       = importdata('ang_capped_ML_vec_interp3.mat');
%projections = projections(:,:, 1:3:222);
%angles      = angles(1:3:222,:);
 
projections  = importdata([pwd, '/data/vector_projections.mat']);
angles       = importdata([pwd, '/data/angles.mat']);


dtype='single';
custom_euler_beam = {[0 0 1], [0 1 0], [1 0 0]}; %rotation axes of beam
beamProp = [0 0 1]; %x ray propagation direction

%% Rotation matrices
%projections = double(projections(:,:,1:178) ); %(:,:,1:178)
%angles = double( angles(1:178,:) ); %(1:178,:)
%%good version, projections should have rotation axis at n/2+1 and this
[dimx, dimy, Num_pj] = size(projections);

Rs = zeros(3,3,Num_pj, dtype);
alphas = zeros(3, Num_pj, dtype);
for k = 1:1:  Num_pj
    phi   = angles(k,1);
    theta = angles(k,2);
    psi   = angles(k,3);
    pj    = projections(:,:,k);    
    
    mat1 = MatrixQuaternionRot([0,0,1], phi);
    mat2 = MatrixQuaternionRot([0,1,0], theta);
    mat3 = MatrixQuaternionRot([1,0,0], psi);
    R = (mat1*mat2*mat3)';
    Rs(:,:,k) = single(R);
    

    
    % beam coefficient
    mat1 = MatrixQuaternionRot([0,0,1], phi);
    mat2 = MatrixQuaternionRot([0,1,0], theta);
    mat3 = MatrixQuaternionRot([1,0,0], psi);
    R_beam = (mat1*mat2*mat3);
    beamProp_rot = R_beam*beamProp';
    alphas(:,k) = beamProp_rot;
end

%% parameter
step_size      = 1.;  %step_size <=1 but can be larger is sparse
iterations     = 100;
dimz           = dimx;
support0 = ones(480,480,480,'logical');

%%
tic
[recX, recY, recZ] = RT3_vector_1GPU3(single(projections), Rs, alphas, dimz, iterations, step_size, support0);
toc
%% calculate projections
cal_projs = RT3_vector_calculateProjections(recX, recY, recZ, Rs, alphas );

figure(2); img(projections,'measured', cal_projs,'calculated','abs','off','caxis',[-0.00003,0.00003])
figure(3); img(cal_projs,'calculated','abs','off','caxis',[-0.00002,0.00002])


    
    
    
    
    
    
    
    
    
    
    
    
    

