%% Gradient Descent
addpath( [pwd, '\src\'] );
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
 
%projections  = importdata([pwd, '\data\lattice\proj_lattice_3.mat']);
%angles       = importdata([pwd, '\data\lattice\ang_lattice_3.mat']);
%sets = [15:75,104:164];
%projections = double( projections(:,:,sets) );
%angles = double( angles(sets,:) );

projections  = importdata([pwd, '\data\lattice\proj_lattice_noisy.mat']);
angles       = importdata([pwd, '\data\lattice\ang_lattice_noisy.mat']);

dtype='single';
custom_euler_beam = {[0 0 1], [0 1 0], [1 0 0]}; %rotation axes of beam
%%
data = importdata ([pwd,'\data\lattice\meta_lattice_model_randperturb.mat']);
mx = data.m_xn; my = data.m_yn; mz = data.m_zn;
%modl = permute(model,[2 3 1]);
%m_xnc = permute(m_xn,[2 3 1]);
%m_ync = permute(m_yn,[2 3 1]);
%m_znc = permute(m_zn,[2 3 1]);
support = data.model > 3;
%figure(100);img(m_xnc,'mx',m_ync,'my',m_znc,'mz','abs','off')
beamProp = [0 0 1]; %x ray propagation direction
%%

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
step_size      = 2;  %step_size <=1 but can be larger is sparse
iterations     = 150;
dimz           = dimx;
support0 = support>0;

%%
tic
[recX, recY, recZ] = RT3_vector_1GPU3(single(projections), Rs, alphas, dimz, iterations, step_size, support0);
toc

%%
%tic
%[rec1, rec2, rec3] = RT3_vector_1GPU(single(projections), Rs, alphas, dimz, 70, step_size, support0);
%[recX, recY, recZ] = RT3_vector_1GPU2(single(projections), Rs, alphas, dimz, iterations, step_size, support0);
%toc


%% calculate projections
cal_projs = RT3_vector_calculateProjections(recX, recY, recZ, Rs, alphas );
figure(1); img(projections,'measured', cal_projs,'calculated')


%% analysis preparation

r_rec     = sqrt(recX.^2+recY.^2+recZ.^2);
Psi_rec   = atan(recY./recX);
Theta_rec = atan(sqrt(recX.^2+recY.^2)./recZ);
Psi_rec(isnan(Psi_rec))=0; Theta_rec(isnan(Theta_rec))=0;

n1_rec = cos(Psi_rec).*sin(Theta_rec); 
n2_rec = sin(Psi_rec).*sin(Theta_rec); 
n3_rec =               cos(Theta_rec); 
%n1_rec(isnan(n1_rec))=0; n2_rec(isnan(n2_rec))=0; n3_rec(isnan(n3_rec))=0;


m     = sqrt(mx.^2+my.^2+mz.^2);
Psi   = atan(my./mx);                 
Theta = atan(sqrt(mx.^2+my.^2)./mz ); 
Psi(isnan(Psi))=0; Theta(isnan(Theta))=0;

n1 = cos(Psi).*sin(Theta); 
n2 = sin(Psi).*sin(Theta); 
n3 = cos(Theta);           
%n1(isnan(n1))=0; n2(isnan(n2))=0; n3(isnan(n3))=0;

%% Show correlation
xcorX = sum(mx(:).*recX(:)) / (norm(recX(:))*norm(mx(:)));
xcorY = sum(my(:).*recY(:)) / (norm(recY(:))*norm(my(:)));
xcorZ = sum(mz(:).*recZ(:)) / (norm(recZ(:))*norm(mz(:)));
m2 = sqrt(mx.^2+my.^2+mz.^2);
xcorM = sum(m2(:).*r_rec(:)) / ( norm(r_rec(:)) * norm(m2(:)) );
xcorXYZ = sum(mx(:).*recX(:) + ...
              my(:).*recY(:) + ...
              mz(:).*recZ(:)) / ...
    sum( sqrt( (mx(:).^2 + my(:).^2 + mz(:).^2).*  ...
  (recX(:).^2 + recY(:).^2 + recZ(:).^2)));

vec_n = (mx(:).*recX(:) + my(:).*recY(:) + ...
         mz(:).*recZ(:)) ./ ...
 sqrt( (mx(:).^2 + my(:).^2 + mz(:).^2).*  ...
  (recX(:).^2 + recY(:).^2 + recZ(:).^2) );
vec_n(isnan(vec_n))=0;
xcorXYZ2 = sum(vec_n(:))/nnz(vec_n(:));

xcorN = sum(n1(:).*n1_rec(:) + n2(:).*n2_rec(:) + n3(:).*n3_rec(:) ) /...
    sum( sqrt( (n1(:).^2 + n2(:).^2 + n3(:).^2) .* ...
    (n1_rec(:).^2 + n2_rec(:).^2 + n3_rec(:).^2) ));

fprintf('corr(x),corr(y),corr(z) = (%.4f, %.4f, %.4f)\n', xcorX,xcorY,xcorZ)
fprintf('corr((x,y,z)) = %.4f, %.4f\n',xcorXYZ, xcorXYZ2);
fprintf('corr(norm(x,y,z)) = %.4f\n',xcorM);
%fprintf('corr(nVector) = %.4f\n',xcorN);

%%
%{
mat1 = MatrixQuaternionRot([0,0,1], 60);
mat2 = MatrixQuaternionRot([0,1,0], 10);
mat3 = MatrixQuaternionRot([1,0,0], 10);
%R_beam = (mat3*mat2*mat1)
%R_beam^(-1)
R_beam = (mat1*mat2*mat3)
%last column
beamProp_rot = R_beam*beamProp'
%}
%% show FSC
[correlationX, freq]  = FourierShellCorrelate(mx, recX, 20);
[correlationY, freq]  = FourierShellCorrelate(my, recY, 20);
[correlationZ, freq]  = FourierShellCorrelate(mz, recZ, 20);
    
figure(1);
plot(freq,correlationX,'-.',freq,correlationY,'--', freq, correlationZ,'-', 'LineWidth',1.5 );
legend('M_x', 'M_y','M_z');
ylim([0.,1]);
xlim([0.,1]);
xlabel('Spatial frequency (% of Nyquist)');
ylabel('Correlation coefficient');

%% show 1-pixel thick central slices
sl = 50;
recX_central = sum( recX(:,:,sl), 3);
recY_central = sum( recY(:,:,sl), 3);
recZ_central = sum( recZ(:,:,sl), 3);

mx_central = sum( mx(:,:,sl), 3);
my_central = sum( my(:,:,sl), 3);
mz_central = sum( mz(:,:,sl), 3);

max_i = max( [recX_central(:);recY_central(:); recZ_central(:); mx_central(:); my_central(:); mz_central(:)] );
min_i = min( [recX_central(:);recY_central(:); recZ_central(:); mx_central(:); my_central(:); mz_central(:)] );

figure(11); img( recX_central, 'abs','off', 'caxis',[ min_i,max_i] )
figure(12); img( recY_central, 'abs','off', 'caxis',[ min_i,max_i] )
figure(13); img( recZ_central, 'abs','off', 'caxis',[ min_i,max_i] )
    
figure(14); img( mx_central, 'abs','off', 'caxis',[ min_i,max_i] )
figure(15); img( my_central, 'abs','off', 'caxis',[ min_i,max_i] )
figure(16); img( mz_central, 'abs','off', 'caxis',[ min_i,max_i] )

%% Show 5-pixel thick central slices
recX_cenXZ = squeeze( sum( recX(:,48:52,:), 2) );
recY_cenXZ = squeeze( sum( recY(:,48:52,:), 2) );
recZ_cenXZ = squeeze( sum( recZ(:,48:52,:), 2) );

mx_cenXZ = squeeze( sum( mx(:,48:52,:), 2) );
my_cenXZ = squeeze( sum( my(:,48:52,:), 2) );
mz_cenXZ = squeeze( sum( mz(:,48:52,:), 2) );

max_i = max( [recX_cenXZ(:);recY_cenXZ(:); recZ_cenXZ(:); mx_cenXZ(:); my_cenXZ(:); mz_cenXZ(:)] );
min_i = min( [recX_cenXZ(:);recY_cenXZ(:); recZ_cenXZ(:); mx_cenXZ(:); my_cenXZ(:); mz_cenXZ(:)] );

figure(21); img( recX_cenXZ, 'abs','off', 'caxis',[ min_i,max_i] )
figure(22); img( recY_cenXZ, 'abs','off', 'caxis',[ min_i,max_i] )
figure(23); img( recZ_cenXZ, 'abs','off', 'caxis',[ min_i,max_i] )
    
figure(24); img( mx_cenXZ, 'abs','off', 'caxis',[ min_i,max_i] )
figure(25); img( my_cenXZ, 'abs','off', 'caxis',[ min_i,max_i] )
figure(26); img( mz_cenXZ, 'abs','off', 'caxis',[ min_i,max_i] )
 
 
    
 %%
 %save('results\rec_sim_metalattice.mat','recX','recY','recZ')
    

