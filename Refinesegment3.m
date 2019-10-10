function tempnewphi2 = Refinesegment3(newphi)

% tempnewphi = newphi;
% se = ones(3,3,3);
% tempnewphi = padarray(tempnewphi,[1 1 1],0,'pre');
% tempnewphi = padarray(tempnewphi,[1 1 1],0,'post');
% 
% tempnewphi = imdilate(tempnewphi,se);
% tempnewphi = imerode(tempnewphi,se);
% 
% tempnewphi = imerode(tempnewphi,se);
% tempnewphi = imdilate(tempnewphi,se);
% 
% tempnewphi2 = tempnewphi(2:end-1,2:end-1,2:end-1);
tempnewphi2 = zeros(size(newphi));
se = ones(10,10);
for slice = 1:size(newphi,3)

  img =  newphi(:,:,slice);
  tempimg = padarray(img,[1 1],0,'pre');
  tempimg = padarray(tempimg,[1 1],0,'post');
  
  tempimg = medfilt2(tempimg,[3,3]);
     
  tempimg = imdilate(tempimg,se);
  tempimg = imfill(tempimg,'holes');
  tempimg = imerode(tempimg,se);


  tempimg2 = tempimg(2:end-1,2:end-1);
  tempnewphi2 (:,:,slice) = tempimg2;
end



%     end

    [L,num] = bwlabeln(tempnewphi2,6);
    % 属于肿瘤的部分必定是像素数最多的部分
    count = length(L(L==1));
    index = 1;
    for i = 2:num
        if length(L(L==i)) > count
            count = length(L(L==i));
            index = i;
        end
    end
    % 将其他部分的mask置为0
    tempnewphi2(find(L~=index)) = 0;
% %     dif = tempnewphi - tempnewphi2;

%  for slice = 1:size(tempnewphi2,1)
%        img = tempnewphi2(slice,:,:);
%        img = reshape(img,[size(img,2) size(img,3)]);
%        
%       tempimg = padarray(img,[1 1],0,'pre');
%       tempimg = padarray(tempimg,[1 1],0,'post');
% 
%       tempimg = medfilt2(tempimg,[3,3]);
% 
%       tempimg = imdilate(tempimg,se);
%       tempimg = imfill(tempimg,'holes');
%       tempimg = imerode(tempimg,se);
% 
%        img = tempimg(2:end-1,2:end-1);
%          
%        img = imfill(img,'holes');
%        img = reshape(img,[1 size(img,1) size(img,2)]);
%        tempnewphi2(slice,:,:) = img;
%  end   
%  
%  
%  
%  for slice = 1:size(tempnewphi2,2)
%        img = tempnewphi2(:,slice,:);
%        img = reshape(img,[size(img,1) size(img,3)]);
%        
%        tempimg = padarray(img,[1 1],0,'pre');
%       tempimg = padarray(tempimg,[1 1],0,'post');
% 
%       tempimg = medfilt2(tempimg,[3,3]);
% 
%       tempimg = imdilate(tempimg,se);
%       tempimg = imfill(tempimg,'holes');
%       tempimg = imerode(tempimg,se);
% 
%        img = tempimg(2:end-1,2:end-1);
%        
%        
%        img = imfill(img,'holes');
%        img = reshape(img,[1 size(img,1) size(img,2)]);
%        tempnewphi2(:,slice,:) = img;
%  end   
%  
 
 
 for slice = 1:size(tempnewphi2,3)
       img = tempnewphi2(:,:,slice);
       img = imfill(img,'holes');
       tempnewphi2(:,:,slice) = img;
 end   
 

end
