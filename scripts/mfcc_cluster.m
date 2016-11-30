% Trial code for 4 speakers

[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\train\dr1\fcjf0\sa1.wav');
speech1=speech(16001:32000);
MFCC1=mfcc_run(speech1,fs);
MFCC1=MFCC1';
MFCC1=[MFCC1 ones(118,1)];
[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\train\dr1\fdaw0\sa1.wav');
speech2=speech(16001:32000);
MFCC2=mfcc_run(speech2,fs);
MFCC2=MFCC2';
MFCC2=[MFCC2 2*ones(118,1)];
[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\train\dr1\fdml0\sa1.wav');
speech2=speech(16001:32000);
MFCC3=mfcc_run(speech2,fs);
MFCC3=MFCC3';
MFCC3=[MFCC3 3*ones(118,1)];    
[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\train\dr1\fecd0\sa1.wav');
speech2=speech(16001:32000);
MFCC4=mfcc_run(speech2,fs);
MFCC4=MFCC4';
MFCC4=[MFCC4 4*ones(118,1)];
MFCC=[MFCC1 ; MFCC2 ; MFCC3 ; MFCC4];
for i=1:size(MFCC,1)
MFCC(i,1:13)=MFCC(i,1:13)/max(abs(MFCC(i,1:13)));
end

% SVM

%t = templateSVM('KernelFunction','polynomial','PolynomialOrder',3);
%Mdl = fitcecoc(MFCC(:,1:13),MFCC(:,14),'Learners',t);

% k-NN

Mdl=fitcknn(MFCC(:,1:13),MFCC(:,14),'NumNeighbors',40);


%Test

[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\test\dr1\faks0\sa1.wav');
speech2=speech(16001:32000);
m_test1=mfcc_run(speech2,fs);
m_test1=m_test1';
m_test1=[m_test1 ones(118,1)];
[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\test\dr1\fdac1\sa1.wav');
speech2=speech(16001:32000);
m_test2=mfcc_run(speech2,fs);
m_test2=m_test2';
m_test2=[m_test2 2*ones(118,1)];
[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\test\dr1\felc0\sa1.wav');
speech2=speech(16001:32000);
m_test3=mfcc_run(speech2,fs);
m_test3=m_test3';
m_test3=[m_test3 3*ones(118,1)];
[speech,fs]=audioread('C:\Users\Rama\Desktop\Sem7\speech\timit\test\dr1\fjem0\sa1.wav');
speech2=speech(16001:32000);
m_test4=mfcc_run(speech2,fs);
m_test4=m_test4';
m_test4=[m_test4 4*ones(118,1)];
m_test=[m_test1; m_test2; m_test3; m_test4];
for i=1:size(m_test,1)
m_test(i,1:13)=m_test(i,1:13)/max(abs(m_test(i,1:13)));
end


label=predict(Mdl,m_test(:,1:13));
sum(label==m_test(:,14))