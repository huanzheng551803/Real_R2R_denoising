mkdir('./srgb_1/')
names = dir (['../../Data/**/*GT_RAW*.MAT']);
length = size(names);
addpath(genpath('../../Data/'))
disp(names)
ne = csvread('/home/huanzheng/Documents/others_work/denoising_data/sidd_train/SIDD_Medium_Raw/noise_level_functions.csv', 1, 1);
for i = 1:10 % length(1)

    gt_name = names(i).name;
    noisy_name = strrep(gt_name, 'GT','NOISY');
    meta_name = strrep(gt_name, 'GT','METADATA');

    gtRAW = load(gt_name);
    gtRAW = gtRAW.x;
    noisyRAW = load(noisy_name);
    noisyRAW = noisyRAW.x;

    metadata = load(meta_name);
    metadata = metadata.metadata;

    img_index = floor((i-1)/2)+1;
    beta = ne(img_index,:);

    [ cfaidx, cfastr ] = cfa_pattern(metadata);
    sigma1 = ones(size(gtRAW));

    sigma1(1:2:end,1:2:end) = beta(cfaidx(1)*2+1)*noisyRAW(1:2:end,1:2:end)+beta(cfaidx(1)*2+2);
    sigma1(1:2:end,2:2:end) = beta(cfaidx(2)*2+1)*noisyRAW(1:2:end,2:2:end)+beta(cfaidx(2)*2+2);
    sigma1(2:2:end,1:2:end) = beta(cfaidx(3)*2+1)*noisyRAW(2:2:end,1:2:end)+beta(cfaidx(3)*2+2);
    sigma1(2:2:end,2:2:end) = beta(cfaidx(4)*2+1)*noisyRAW(2:2:end,2:2:end)+beta(cfaidx(4)*2+2);
    noise = normrnd(0,1,size(gtRAW,1),size(gtRAW,2)).*sqrt(sigma1);

    gtSrgb = run_pipeline(gtRAW, metadata, 'normal', 'srgb');
    %// imwrite(gtSrgb,  fullfile('./srgb_1', strrep(gt_name, '.MAT','_srgb_gt.png')));

    noisySrgb = run_pipeline(noisyRAW, metadata, 'normal', 'srgb');
    %// imwrite(noisySrgb, fullfile('./srgb_1',strrep(gt_name, '.MAT','_srgb_noise.png')));

    r2r1Srgb = run_pipeline(noisyRAW+noise, metadata, 'normal', 'srgb');
    %// imwrite(r2r1Srgb, fullfile('./srgb_1',strrep(gt_name, '.MAT','_srgb_r2r1.png')));

    r2r2Srgb = run_pipeline(noisyRAW-noise, metadata, 'normal', 'srgb');
    %// imwrite(r2r2Srgb, fullfile('./srgb_1',strrep(gt_name, '.MAT','_srgb_r2r2.png')));

    a = mean(abs(r2r2Srgb+r2r1Srgb-2*noisySrgb),'all')
    %// noise = normrnd(0,1,size(gtRAW,1),size(gtRAW,2)).*sqrt(sigma1);
    %// r2r1Srgb = run_pipeline(noisyRAW+noise*2, metadata, 'normal', 'srgb');
    %// imwrite(r2r1Srgb, fullfile('./srgb_1',strrep(gt_name, '.MAT','_srgb_r2r1_1.png')));
    %// noise = normrnd(0,1,size(gtRAW,1),size(gtRAW,2)).*sqrt(sigma1);
    %// r2r1Srgb = run_pipeline(noisyRAW+noise*2, metadata, 'normal', 'srgb');
    %// imwrite(r2r1Srgb, fullfile('./srgb_1',strrep(gt_name, '.MAT','_srgb_r2r1_2.png')));
    disp([gt_name ' ' num2str(psnr(noisySrgb,gtSrgb)) ' ' num2str(psnr(r2r1Srgb,gtSrgb)) ' ' num2str(psnr(r2r2Srgb,gtSrgb)) ' '])
end