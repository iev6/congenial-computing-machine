% Run this after getting the train and test mfcc vectors

[m,v,w]=gaussmix(MFCC(:,1:13),[],[],4,'v');

gmm=gmdistribution(m,v,w);

label = cluster(gmm,m_test(:,1:13));

sum(label==m_test(:,14))

