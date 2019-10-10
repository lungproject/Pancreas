clc;
clear;
str1 = 'X:\Work\Data\yushi\1-234shengjing\';
filestruct=dir(fullfile(str1,'*.*'));
filestruct=filestruct([3 6]);
filenum = size(filestruct,1); 

for count =1:filenum%;18  28
    
    data1=filestruct(count).name;
    filestruct2=dir(fullfile(strcat(str1,data1),'*.*'));
    data2 =  filestruct2(3).name
    filestruct3=dir(fullfile(strcat(str1,data1,'\',data2),'*.*'));
    data3 =  filestruct3(3).name;
    filestruct4=dir(fullfile(strcat(str1,data1,'\',data2,'\',data3),'*.*'));
    datart =  filestruct4(4).name;
    datact =  filestruct4(3).name;
    
    pathct = strcat(str1,data1,'\',data2,'\',data3,'\',datact,'\ct\');
    
    filestruct5=dir(fullfile(strcat(str1,data1,'\',data2,'\',data3,'\',datart,'\RTSTRUCT\')));
    pathrt = strcat(str1,data1,'\',data2,'\',data3,'\',datart,'\RTSTRUCT\',filestruct5(3).name);
    
    [datasetInfo, dataImages, thedirectory, zdist] = dicomloaddataset( pathct );
    dataImages = convertImageDataToHU ( dataImages, datasetInfo );
    [contourList, ~, ~] = dicomReadRT(pathrt );
    
    i = 1;
    while i <= size(contourList,1)
        if ~strcmp(datasetInfo(1).FrameOfReferenceUID, contourList(i).referencedFrameOfReferenceUID)
            contourList(i,:) = []; 
        else
            i = i + 1;
        end
    end

    for i=1:size(contourList,1)
    %     Because Rib Cage has issues displaying, skip rib cage
    if strcmp(contourList(i).structureName, 'Rib Cage'), continue; end;

        %transform to grid points
        contourList(i).pointsList = transformPoints_PatientToGrid(datasetInfo, contourList(i).pointsList);
        %close contour
        contourList(i).pointsList = closeContours( contourList(i).pointsList );
    end

%     f = figure3d(dataImages, datasetInfo, contourList);
    if strcmp(contourList(3).color,'27\205\26')
        noduleMask = createMaskFromContour( dataImages, contourList(3).pointsList );
    elseif strcmp(contourList(2).color,'27\205\26')
        noduleMask = createMaskFromContour( dataImages, contourList(2).pointsList );
    else
        noduleMask = createMaskFromContour( dataImages, contourList(1).pointsList );
    end
    
    save(strcat('X:\Work\Data\yushi\Allmat\',data2,'_2.mat'),'dataImages','noduleMask');

end

clc;
clear;
str1 = 'X:\Work\Data\yushi\235-404Yang\';
filestruct=dir(fullfile(str1,'*.*'));
filenum = size(filestruct,1); 

for count = 3:filenum%;18  28
    
    data1=filestruct(count).name;
    filestruct2=dir(fullfile(strcat(str1,data1),'*.*'));
    data2 =  filestruct2(3).name
    filestruct3=dir(fullfile(strcat(str1,data1,'\',data2),'*.*'));
    data3 =  filestruct3(3).name;
    filestruct4=dir(fullfile(strcat(str1,data1,'\',data2,'\',data3),'*.*'));
    ttemp = filestruct4(4).name;
    if strcmp(ttemp(1:2),'HM')
        datart =  filestruct4(4).name;
        datact =  filestruct4(3).name;
    else
        datart =  filestruct4(3).name;
        datact =  filestruct4(4).name;
    end
    
    pathct = strcat(str1,data1,'\',data2,'\',data3,'\',datact,'\ct\');
    
    filestruct5=dir(fullfile(strcat(str1,data1,'\',data2,'\',data3,'\',datart,'\RTSTRUCT\')));
    pathrt = strcat(str1,data1,'\',data2,'\',data3,'\',datart,'\RTSTRUCT\',filestruct5(3).name);
    
    [datasetInfo, dataImages, thedirectory, zdist] = dicomloaddataset( pathct );
    dataImages = convertImageDataToHU ( dataImages, datasetInfo );
    [contourList, ~, ~] = dicomReadRT(pathrt );
    
    i = 1;
    while i <= size(contourList,1)
        if ~strcmp(datasetInfo(1).FrameOfReferenceUID, contourList(i).referencedFrameOfReferenceUID)
            contourList(i,:) = []; 
        else
            i = i + 1;
        end
    end

    for i=1:size(contourList,1)
    %     Because Rib Cage has issues displaying, skip rib cage
    if strcmp(contourList(i).structureName, 'Rib Cage'), continue; end;

        %transform to grid points
        contourList(i).pointsList = transformPoints_PatientToGrid(datasetInfo, contourList(i).pointsList);
        %close contour
        contourList(i).pointsList = closeContours( contourList(i).pointsList );
    end

%     f = figure3d(dataImages, datasetInfo, contourList);
    if strcmp(contourList(3).color,'27\205\26')
        noduleMask = createMaskFromContour( dataImages, contourList(3).pointsList );
    elseif strcmp(contourList(2).color,'27\205\26')
        noduleMask = createMaskFromContour( dataImages, contourList(2).pointsList );
    else
        noduleMask = createMaskFromContour( dataImages, contourList(1).pointsList );
    end
    
    save(strcat('X:\Work\Data\yushi\Allmat\',data2,'.mat'),'dataImages','noduleMask');

end


clc;
clear;
str1 = 'X:\Work\Data\yushi\405-517RTfilesLuhong60\';
filestruct=dir(fullfile(str1,'*.*'));
filenum = size(filestruct,1); 

for count = 55:filenum%;18  28
    
    data1=filestruct(count).name;
    filestruct2=dir(fullfile(strcat(str1,data1),'*.*'));
    data2 =  filestruct2(3).name
    filestruct3=dir(fullfile(strcat(str1,data1,'\',data2),'*.*'));
    data3 =  filestruct3(3).name;
    filestruct4=dir(fullfile(strcat(str1,data1,'\',data2,'\',data3),'*.*'));
    ttemp = filestruct4(4).name;
    str ='HM';
    if strcmp(ttemp(1:min(2,length(ttemp))),str(1:min(2,length(ttemp))))
        datart =  filestruct4(4).name;
        datact =  filestruct4(3).name;
    else
        datart =  filestruct4(3).name;
        datact =  filestruct4(4).name;
    end
    
    pathct = strcat(str1,data1,'\',data2,'\',data3,'\',datact,'\ct\');
    
    filestruct5=dir(fullfile(strcat(str1,data1,'\',data2,'\',data3,'\',datart,'\RTSTRUCT\')));
    pathrt = strcat(str1,data1,'\',data2,'\',data3,'\',datart,'\RTSTRUCT\',filestruct5(3).name);
    
    [datasetInfo, dataImages, thedirectory, zdist] = dicomloaddataset( pathct );
    dataImages = convertImageDataToHU ( dataImages, datasetInfo );
    [contourList, ~, ~] = dicomReadRT(pathrt );
    
    i = 1;
    while i <= size(contourList,1)
        if ~strcmp(datasetInfo(1).FrameOfReferenceUID, contourList(i).referencedFrameOfReferenceUID)
            contourList(i,:) = []; 
        else
            i = i + 1;
        end
    end

    for i=1:size(contourList,1)
    %     Because Rib Cage has issues displaying, skip rib cage
    if strcmp(contourList(i).structureName, 'Rib Cage'), continue; end;

        %transform to grid points
        contourList(i).pointsList = transformPoints_PatientToGrid(datasetInfo, contourList(i).pointsList);
        %close contour
        contourList(i).pointsList = closeContours( contourList(i).pointsList );
    end

%     f = figure3d(dataImages, datasetInfo, contourList);
    if strcmp(contourList(3).color,'27\205\26')
        noduleMask = createMaskFromContour( dataImages, contourList(3).pointsList );
    elseif strcmp(contourList(2).color,'27\205\26')
        noduleMask = createMaskFromContour( dataImages, contourList(2).pointsList );
    else
        noduleMask = createMaskFromContour( dataImages, contourList(1).pointsList );
    end
    
    save(strcat('X:\Work\Data\yushi\Allmat\',data2,'.mat'),'dataImages','noduleMask');

end


