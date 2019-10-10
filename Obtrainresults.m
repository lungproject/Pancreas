close all
clear
testfeature = load('./Results/predicttest.txt');
load('./data/patientsind.mat');

pretest = testfeature(:,2);
testdata = unique(Grouptrain);
for i=1:length(testdata)
    ind =  find(Grouptrain==testdata(i)); 
    temp1 = pretest(ind,:);
    testpp(i,:)=mean(temp1); 
   

end

