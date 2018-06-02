% M = readtable('/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/code/Annotations.csv')
M = fopen('/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/code/test.txt');
ref_annotation = [[1236.0, 96.0]; [791.0, 234.0]; [641.0, 1023.0]; [660.0, 1419.0]; [1855.0, 1434.0]; [1880.0, 1058.0]; [1682.0, 198.0]; [1235.0, 526.0]; [1244.0, 835.0]]; % landmarks or annotation of the image that has to be warped
for i=1:102
    line_ex = fgetl(M)
    splited = strsplit(line_ex, '*');
    img_name = splited{1};
    for j = 2:19
       a(j-1) = str2double(splited{j});
    end
    img_annotation = transpose(reshape(a, [2,9]));
    img = imread(strcat('/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/images/', img_name, '.jpg'));
    img_gray = rgb2gray(img); % Convert the input image to gray scale
    [imo,mask] = rbfwarp2d(img_gray,  img_annotation, ref_annotation, 'thin'); % Warp the input image and save it in imo
    imwrite(uint8(imo), strcat('/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/warped_images/', img_name, '.jpg')); % save the imo as .jpg image.
end
% img_annotation = [[1275.0, 44.0]; [841.0, 175.0]; [626.0, 1023.0]; [685.0, 1327.0]; [1795.0, 1396.0]; [1897.0, 1069.0]; [1716.0, 171.0]; [1263.0, 413.0]; [1275.0, 720.0]]; % landmarks or annotations of the reference image
% img_gray = rgb2gray(img); % Convert the input image to gray scale
% [imo,mask] = rbfwarp2d(img_gray,  img_annotation, ref_annotation, 'thin'); % Warp the input image and save it in imo
% imwrite(uint8(imo), '/home/cougarnet.uh.edu/amobiny/Desktop/output.jpg'); % save the imo as .jpg image.