clc;
clear;
filestruct=dir(fullfile('./data/','*.mat'));
filenum = size(filestruct,1); 

labels = [0 1];
count=1;
for datanum = 1:filenum
    data1=filestruct(datanum).name;
    path = strcat('./data/',data1);
    load(path);

    noduleMask(isnan(noduleMask))=0;
    noduleMask(noduleMask~=0)=1;
    noduleMask = Refinesegment3(noduleMask);
    [image,mask] = ROIpart_2d2(dataImages,noduleMask);
    
    chestwl = 50;
    chestww = 350;
    image = mat2gray(image,[chestwl-0.5*chestww,chestwl+0.5*chestww]);   
        
    nodule = image.*mask;
    
    for slicenum=1:size(mask,3)
        
         
         [nodulepatch,maskpatch] = ROIpart2_2d2(nodule(:,:,slicenum),mask(:,:,slicenum));    
         
         nodulepatch = imresize(nodulepatch,[64 64],'nearest');
         maskpatch = imresize(maskpatch,[64 64],'nearest');
         maskpatch = round(maskpatch);

         nodulepatch = (nodulepatch-mean(nodulepatch(:)))/std(nodulepatch(:));   
        
         trainpatch(:,:,count) = nodulepatch;            
         Grouptrain(count,1) = datanum;
         count=count+1;
    end
       
end  
trainpatch = permute(trainpatch,[3 1 2]);
writeNPY(trainpatch, './data/sample_64.npy');
save('./data/patientsind.mat','Grouptrain');

    