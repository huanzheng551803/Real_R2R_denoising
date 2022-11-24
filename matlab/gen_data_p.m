mkdir('./srgb/')
names = dir (['../../Data/**/*GT_RAW*.MAT']);
length = size(names);
addpath(genpath('../../Data/'))
disp(names)
ne = csvread('/home/huanzheng/Documents/others_work/denoising_data/sidd_train/SIDD_Medium_Raw/noise_level_functions.csv', 1, 1);
for i = 1:length(1)

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
    sigma1(1:2:end,1:2:end) =beta(cfaidx(1)*2+2);
    sigma1(1:2:end,2:2:end) =beta(cfaidx(2)*2+2);
    sigma1(2:2:end,1:2:end) =beta(cfaidx(3)*2+2);
    sigma1(2:2:end,2:2:end) =beta(cfaidx(4)*2+2);

    sigma2 = ones(size(gtRAW));
    sigma2(1:2:end,1:2:end) = gtRAW(1:2:end,1:2:end)/beta(cfaidx(1)*2+1);
    sigma2(1:2:end,2:2:end) = gtRAW(1:2:end,2:2:end)/beta(cfaidx(2)*2+1);
    sigma2(2:2:end,1:2:end) = gtRAW(2:2:end,1:2:end)/beta(cfaidx(3)*2+1);
    sigma2(2:2:end,2:2:end) = gtRAW(2:2:end,2:2:end)/beta(cfaidx(4)*2+1);



    noise = normrnd(0,1,size(gtRAW,1),size(gtRAW,2)).*sqrt(sigma1)+poissrnd(sigma2).*(gtRAW./sigma2)-gtRAW;
    gtSrgb = run_pipeline(gtRAW, metadata, 'normal', 'srgb');
    imwrite(gtSrgb,  fullfile('./srgb1', strrep(gt_name, '.MAT','_srgb_gt.png')));

    noisySrgb = run_pipeline(noisyRAW, metadata, 'normal', 'srgb');
    imwrite(noisySrgb, fullfile('./srgb1',strrep(gt_name, '.MAT','_srgb_noise.png')));

    r2r1Srgb = run_pipeline(noisyRAW+noise, metadata, 'normal', 'srgb');
    imwrite(r2r1Srgb, fullfile('./srgb',strrep(gt_name, '.MAT','_srgb_r2r1.png')));

    r2r2Srgb = run_pipeline(noisyRAW-noise, metadata, 'normal', 'srgb');
    imwrite(r2r2Srgb, fullfile('./srgb1',strrep(gt_name, '.MAT','_srgb_r2r2.png')));

    n2nSrgb = run_pipeline(gtRAW+noise, metadata, 'normal', 'srgb');
    % imwrite(r2r2Srgb, fullfile('./srgb1',strrep(gt_name, '.MAT','_srgb_r2r2.png')));

    noise = normrnd(0,1,size(gtRAW,1),size(gtRAW,2)).*sqrt(sigma1)+poissrnd(sigma2).*(gtRAW./sigma2)-gtRAW;
    r2r1Srgb = run_pipeline(noisyRAW+noise, metadata, 'normal', 'srgb');
    imwrite(r2r1Srgb, fullfile('./srgb1',strrep(gt_name, '.MAT','_srgb_r2r1_1.png')));
    noise = normrnd(0,1,size(gtRAW,1),size(gtRAW,2)).*sqrt(sigma1)+poissrnd(sigma2).*(gtRAW./sigma2)-gtRAW;
    r2r1Srgb = run_pipeline(noisyRAW+noise, metadata, 'normal', 'srgb');
    imwrite(r2r1Srgb, fullfile('./srgb1',strrep(gt_name, '.MAT','_srgb_r2r1_2.png')));
    disp([gt_name ' ' num2str(psnr(noisySrgb,gtSrgb)) ' ' num2str(psnr(r2r1Srgb,gtSrgb)) ' ' num2str(psnr(r2r2Srgb,gtSrgb)) ' ' num2str(psnr(n2nSrgb,gtSrgb)) ])
end