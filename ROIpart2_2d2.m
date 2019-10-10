function [part1,seg1] = ROIpart2_2d2(Image1,Segmentation1)

     [m,n]=size(Image1);   
    Segmentation = Segmentation1;
    Segmentation(Segmentation~=0)=1;
    [x,y] = ind2sub(size(Segmentation),find(Segmentation~=0));
    xrange = max(min(x),1):min(max(x),size(Segmentation,1));
    yrange = max(min(y),1):min(max(y),size(Segmentation,2));    


    num= max([length(xrange) length(yrange)]);
     if length(xrange)~=num
         dif = num-length(xrange);
         if mod(dif,2)~=0
             difl = (dif-1)/2;
             difr = dif-difl;
         else
             difl = dif/2;
             difr = dif/2;
         end
        
        lrange = min(xrange)-difl;
        numl=[];numr=[];
        if lrange<1
            numl= ones(1,difl-(min(xrange)-1));
            lrange=1;
            
        end

        rrange = max(xrange)+difr;
        if rrange>m
            numr= m*ones(1,rrange-m);
            rrange=m;
            
        end

        xrange = [numl lrange:rrange numr];
     end

     if length(yrange)~=num
         dif = num-length(yrange);
         if mod(dif,2)~=0
             difl = (dif-1)/2;
             difr = dif-difl;
         else
             difl = dif/2;
             difr = dif/2;
         end
         
        lrange = min(yrange)-difl;
        numl=[];numr=[];
        if lrange<1
            numl= ones(1,difl-(min(yrange)-1));
            lrange=1;
            
        end

        rrange = max(yrange)+difr;
        if rrange>n
            numr= n*ones(1,rrange-n);
            rrange=n;            
        end

        yrange = [numl lrange:rrange numr];
     end
      
     dis=5;
     xrange = max(xrange(1)-dis,1):min(xrange(end)+dis,size(Image1,1));%250
     yrange = max(yrange(1)-dis,1):min(yrange(end)+dis,size(Image1,2));
     
     part1 = Image1(xrange,yrange);
     seg1 = Segmentation1(xrange,yrange);
     